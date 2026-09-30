# Setting up the BugFlow GitHub App

This connects a real GitHub repository to BugFlow so opening or updating a
pull request automatically gets a risk check and a comment. **This step is
optional** — everything in Phase 4 also works fully offline via
`scripts/replay_pr_events.py` and the built-in "fake Checks sink"
(`app.integrations.github.client.FakeGitHubClient`), which is what runs by
default until you complete this guide.

## 1. Register the App

1. GitHub → Settings → Developer settings → **GitHub Apps** → **New GitHub App**.
2. **Homepage URL:** your BugFlow instance's URL (anything reachable is fine
   for a student deployment, e.g. the ngrok/tunnel URL below).
3. **Webhook URL:** `https://<your-public-url>/webhooks/github`
   - Locally, expose your API with a tunnel (e.g. `ngrok http 8000`) and use
     the tunnel's HTTPS URL — GitHub requires a public, HTTPS endpoint.
4. **Webhook secret:** generate one (`openssl rand -hex 32`) and note it
   down — it goes in `.env` as `GITHUB_WEBHOOK_SECRET` in step 4 below.

### Permissions — least privilege (C6/NFR-US-10)

Grant **only**:

| Permission | Access | Why |
|---|---|---|
| Contents | Read-only | Not currently used to fetch diffs (we use the Commits API instead), kept read-only for future line-level features (Phase 5) |
| Pull requests | Read-only | To receive `pull_request` events |
| Metadata | Read-only | Mandatory baseline permission for every GitHub App |
| Checks | Read & write | To create/update the risk status check |
| Issues | Read & write | Pull request comments are, under the hood, issue comments |

Do **not** grant anything else (no Actions, no Administration, no write
access to Contents) — the app never pushes code or changes settings.

### Subscribe to events

Only **Pull request**.

## 2. Generate a private key

On the App's settings page, scroll to **Private keys** → **Generate a
private key**. This downloads a `.pem` file — treat it like any other
credential (C6: never commit it).

## 3. Install the App

**Install App** → choose the repository (or repositories) you want BugFlow
scoring. Note the **installation ID** from the URL after installing
(`https://github.com/settings/installations/<installation_id>`).

## 4. Configure BugFlow

In `.env` (never committed — see `.env.example`):

```bash
GITHUB_APP_ID=<App ID, shown on the app's settings page>
GITHUB_APP_PRIVATE_KEY_PATH=/path/to/the-downloaded-key.pem
GITHUB_WEBHOOK_SECRET=<the secret from step 1>
```

Mount the `.pem` file into the API and worker containers (both need to sign
requests) — e.g. add a volume in `docker-compose.yml`:

```yaml
api:
  volumes:
    - ./secrets/github-app-private-key.pem:/app/secrets/github-app-private-key.pem:ro
worker:
  volumes:
    - ./secrets/github-app-private-key.pem:/app/secrets/github-app-private-key.pem:ro
```

and set `GITHUB_APP_PRIVATE_KEY_PATH=/app/secrets/github-app-private-key.pem`
to match.

Then, in BugFlow itself, register the repository (`POST /repositories` or
the Repositories page) with its GitHub URL. The first webhook BugFlow
receives for that repo automatically fills in `github_installation_id` —
nothing else to configure.

## Verifying it worked

Open a pull request (or push a commit to an existing one) on the connected
repo. Within a few seconds you should see:
- A "BugFlow risk assessment" check appear on the PR, first `pending` then
  resolving to a conclusion
- A comment from the App with the risk level, probability, and explanation

If nothing appears, check the worker's logs
(`docker compose logs worker -f`) — every failure path (C7: service down;
US-08 AC2: no trained model yet) still updates the check to `completed`
with a `neutral` conclusion and an explanatory summary, so a stuck
`pending` check usually means the webhook itself never arrived (check
GitHub's "Recent Deliveries" tab on the App's settings page for the
delivery status and response code).

## Merge blocking (US-47)

Off by default for every repository. To require a passing risk check before
merging, an Admin can flip `merge_blocking_enabled` on the repository (via
`PATCH /repositories/{id}`) **and** add "BugFlow risk assessment" as a
required status check in the repository's own GitHub branch protection
rules — BugFlow only controls what the check *reports*, GitHub's branch
protection is what actually blocks the merge button.
