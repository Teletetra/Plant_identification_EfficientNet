# Agent Rules: Implementation Permission Requirement

To ensure full user control over any code modifications, the agent must adhere to the following rules:

1. **Ask for Explicit Permission**: Before starting any code implementation, modifications, or writing files, the agent MUST present a clear plan and obtain explicit user permission/approval.
2. **Planning Mode Enforcement**:
   - The agent should always operate in a planning-first mindset.
   - For all features, fixes, or changes, the agent must first draft or update an `implementation_plan.md` artifact.
   - The agent must stop and wait for the user to approve the plan (via the "Proceed" button or explicit text approval) before executing any file changes, running build tasks, or starting implementation.
3. **No Auto-Proceed**: The agent should never assume auto-proceed or bypass user consent for code execution, file changes, or installations unless specifically requested or pre-authorized by the user.
