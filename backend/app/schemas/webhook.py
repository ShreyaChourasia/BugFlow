from pydantic import BaseModel


class WebhookRef(BaseModel):
    sha: str


class WebhookPullRequest(BaseModel):
    number: int
    title: str
    head: WebhookRef
    base: WebhookRef


class WebhookRepository(BaseModel):
    full_name: str
    html_url: str


class WebhookInstallation(BaseModel):
    id: int


class PullRequestWebhookEvent(BaseModel):
    action: str
    pull_request: WebhookPullRequest
    repository: WebhookRepository
    installation: WebhookInstallation | None = None
