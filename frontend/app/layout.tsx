import type { Metadata } from "next";
import "./globals.css";

import { Nav } from "./nav";

export const metadata: Metadata = {
  title: "BugFlow",
  description: "A software quality assistant for pre-merge risk and bug triage.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-border px-6 py-4 flex items-center justify-between">
            <span className="font-semibold text-lg">BugFlow</span>
            <Nav />
          </header>
          <main className="flex-1 p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
