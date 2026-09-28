# Frontend Engineering Skills — Next.js + TypeScript + ShadCN

## Purpose

These skills define the standard approach for building and modifying frontend applications using Next.js, React, TypeScript, ShadCN/UI, and related modern frontend technologies.

These rules are **project-agnostic**.

Before writing code, inspect the existing project and follow its established architecture, naming conventions, components, APIs, routing, styling, authentication, permissions, and state-management patterns.

Do not assume a project uses a particular folder structure, API format, permission system, or library unless it is confirmed from the existing codebase.

---

# Skill 1: Repository Inspection

Before implementing any frontend feature:

1. Inspect the project structure.
2. Inspect existing routes.
3. Inspect layouts.
4. Inspect shared components.
5. Inspect theme and design tokens.
6. Inspect API/client utilities.
7. Inspect existing services.
8. Inspect types and schemas.
9. Inspect authentication/session handling.
10. Inspect permission/RBAC implementation if present.
11. Inspect state-management patterns.
12. Inspect existing similar features.
13. Inspect testing setup.
14. Inspect linting and formatting configuration.
15. Identify the closest existing implementation and follow its pattern.

### Rule

Do not invent architecture when an established project pattern already exists.

---

# Skill 2: Frontend Feature Builder

When implementing a feature:

1. Understand the existing architecture.
2. Identify affected routes.
3. Identify required components.
4. Identify required API operations.
5. Define or reuse types.
6. Define validation schemas where required.
7. Create/reuse service functions.
8. Create/reuse form components.
9. Implement list views where required.
10. Implement create flows where required.
11. Implement edit flows where required.
12. Implement details views where required.
13. Add loading states.
14. Add error states.
15. Add empty states.
16. Add permission checks where applicable.
17. Add responsive behavior.
18. Add tests where the project has testing infrastructure.
19. Update documentation when appropriate.

Do not create unnecessary abstractions.

---

# Skill 3: Page Builder

Every page should:

1. Use the existing application layout.
2. Reuse the existing page-header pattern.
3. Reuse shared container/layout components.
4. Follow existing breadcrumb conventions.
5. Follow existing spacing conventions.
6. Follow existing typography hierarchy.
7. Follow existing action placement patterns.
8. Respect authentication requirements.
9. Respect permissions where applicable.
10. Handle loading.
11. Handle errors.
12. Handle empty states where applicable.
13. Handle not-found states where applicable.
14. Work on supported screen sizes.

Do not create a unique page structure when an existing pattern can be reused.

---

# Skill 4: Component Builder

Before creating a component:

1. Search for an existing reusable component.
2. Check whether the requirement can be solved by composition.
3. Follow existing component naming conventions.
4. Keep components focused.
5. Avoid unnecessarily large components.
6. Avoid premature abstraction.
7. Keep business logic out of purely presentational components where practical.
8. Keep reusable components configurable through typed props.
9. Avoid duplicated UI logic.
10. Preserve accessibility.

Prefer composition over duplication.

---

# Skill 5: Form Builder

For every form:

1. Use the project's existing form library/pattern.
2. Use strongly typed form values.
3. Use schema validation where the project supports it.
4. Add field-level validation.
5. Mark required fields clearly.
6. Provide useful labels.
7. Provide helper text where needed.
8. Display validation errors near their fields.
9. Disable submission while saving.
10. Prevent accidental duplicate submissions.
11. Show success feedback.
12. Show user-friendly API errors.
13. Preserve entered values when appropriate.
14. Handle server-side validation errors.
15. Support keyboard navigation and accessibility.
16. Reuse the same form component for create/edit when the fields are substantially the same.

Avoid duplicating create and edit forms.

---

# Skill 6: Table Builder

For list/data-heavy pages:

1. Use the project's shared table/DataTable component.
2. Define typed column definitions.
3. Add search where useful.
4. Add filters where useful.
5. Add sorting where useful.
6. Add pagination where required.
7. Add row actions where appropriate.
8. Respect permissions for actions.
9. Add loading skeletons.
10. Add empty states.
11. Handle API errors.
12. Support responsive layouts.
13. Use horizontal scrolling when wide tables cannot reasonably collapse.
14. Preserve useful table state when navigating where appropriate.

Do not create one-off table implementations if a shared table system exists.

---

# Skill 7: Details Page Builder

For important entities:

1. Display the entity name/title clearly.
2. Display important metadata.
3. Display status using the existing status/badge pattern.
4. Group related information logically.
5. Use cards/sections where consistent with the application.
6. Show related records where useful.
7. Show activity/audit information where available.
8. Show edit actions only when allowed.
9. Handle loading.
10. Handle not found.
11. Handle forbidden.
12. Handle server errors.

Do not overload the details page with information that is not useful to the user.

---

# Skill 8: API Service Builder

API calls should not normally be implemented directly inside UI components.

For API integrations:

1. Inspect the existing API client.
2. Reuse the shared API client.
3. Create a service layer when the project uses one.
4. Define typed request payloads.
5. Define typed responses.
6. Handle response wrappers consistently.
7. Handle authentication errors.
8. Handle authorization errors.
9. Handle validation errors.
10. Handle not-found responses.
11. Handle server errors.
12. Normalize errors using existing project conventions.

Example:

```text
employee.service.ts
product.service.ts
order.service.ts
customer.service.ts
```

Do not invent endpoint paths or HTTP methods. Verify them from the backend/API contract or existing code.

---

# Skill 9: Type Builder

Avoid `any` unless there is a justified and documented reason.

For each feature, define or reuse:

1. Request types.
2. Response types.
3. List item types.
4. Detail types.
5. Form value types.
6. Filter types.
7. Pagination types.
8. Enum/status types where applicable.

Prefer shared types when they are genuinely reusable.

Do not duplicate identical types across multiple files.

---

# Skill 10: Validation Builder

For user input:

1. Identify required fields.
2. Identify data types.
3. Identify allowed ranges.
4. Identify format requirements.
5. Reuse backend/shared schemas when available.
6. Use frontend validation for immediate feedback.
7. Do not assume frontend validation replaces backend validation.
8. Display errors clearly.
9. Handle server-side validation failures separately.

Frontend validation improves UX; backend validation remains authoritative.

---

# Skill 11: Authentication Builder

For protected applications:

1. Follow the existing authentication implementation.
2. Check authentication state before protected content.
3. Handle loading while authentication state is being resolved.
4. Handle expired sessions.
5. Handle logout.
6. Handle unauthorized access.
7. Avoid exposing protected information before authentication is confirmed.
8. Do not invent authentication mechanisms.

If OAuth, SSO, MFA, or another authentication system already exists, integrate with it rather than creating a parallel system.

---

# Skill 12: RBAC / Authorization UI Builder

If the application has permissions/RBAC:

1. Inspect the existing permission model.
2. Reuse existing permission constants/utilities.
3. Identify required permissions for pages.
4. Identify required permissions for actions.
5. Hide inaccessible navigation items.
6. Hide inaccessible actions where appropriate.
7. Handle direct navigation to unauthorized pages.
8. Show an appropriate forbidden state.
9. Keep authorization logic consistent across the application.

### Critical Rule

Frontend permission checks are for UX only.

The backend must remain the source of truth for authorization.

Never invent permission names.

---

# Skill 13: Navigation / Sidebar Builder

When adding navigation:

1. Inspect the existing navigation architecture.
2. Reuse the existing registry/configuration system if present.
3. Add label.
4. Add path.
5. Add icon if the project uses icons.
6. Add required permission if applicable.
7. Add nesting/children where required.
8. Respect feature/module availability if the application supports feature flags.
9. Ensure inaccessible items are hidden or disabled according to existing conventions.

Do not hardcode navigation in multiple places.

---

# Skill 14: Theme & Styling Builder

When styling:

1. Use the project's existing design system.
2. Use ShadCN/UI components where applicable.
3. Use existing CSS variables/design tokens.
4. Reuse existing typography.
5. Reuse existing spacing.
6. Reuse existing radius/border/shadow patterns.
7. Support light/dark mode if the project supports it.
8. Use semantic colors.
9. Avoid random hardcoded colors.
10. Avoid unnecessary gradients.
11. Avoid excessive visual decoration.
12. Maintain consistency between modules.

Do not introduce a new visual language for one feature.

---

# Skill 15: Responsive UI Builder

Every page should be checked across supported breakpoints.

1. Desktop layout should remain usable.
2. Tablet layout should remain usable.
3. Mobile layout should remain usable when mobile is supported.
4. Avoid unnecessary fixed widths.
5. Avoid content overflowing the viewport.
6. Tables should scroll or adapt appropriately.
7. Forms should collapse naturally.
8. Dialogs should fit smaller screens.
9. Navigation should follow the existing responsive pattern.
10. Buttons and interactive elements should remain accessible.

Do not sacrifice desktop usability merely to achieve mobile responsiveness.

---

# Skill 16: Loading State Builder

Every asynchronous UI should have an intentional loading state.

Use:

* Skeletons for content-heavy pages.
* Table skeletons for tables.
* Card skeletons for dashboards.
* Form skeletons for edit pages.
* Button loading states during mutations.

Avoid:

* Empty white screens.
* Layout shifts.
* Flashing incomplete content.
* Blocking the entire application when only one component is loading.

Prefer localized loading states where practical.

---

# Skill 17: Error State Builder

Errors should be understandable and actionable.

Handle separately where applicable:

1. Validation errors.
2. Unauthorized errors.
3. Forbidden errors.
4. Not-found errors.
5. Network errors.
6. Server errors.
7. Unknown errors.

For each:

* Show a user-friendly message.
* Avoid exposing stack traces.
* Provide retry where useful.
* Provide navigation/back actions where useful.
* Preserve useful context where possible.

Do not silently swallow errors.

---

# Skill 18: Empty State Builder

Avoid generic messages such as:

```text
No data.
Nothing found.
Empty.
```

A useful empty state should communicate:

1. What is missing.
2. Why it matters.
3. What the user can do next.
4. A CTA when the user has permission to perform the action.

Example:

```text
No customers found.

Create your first customer to start managing customer records.

[Create Customer]
```

If the user has no permission to create the resource, do not show a misleading CTA.

---

# Skill 19: Create/Edit Flow Builder

When create and edit operations share the same fields:

1. Create one reusable form.
2. Create page supplies default values.
3. Edit page loads existing values.
4. Edit page handles loading.
5. Edit page handles not found.
6. Edit page handles forbidden.
7. Submit action shows progress.
8. API errors map to fields where possible.
9. Success shows appropriate feedback.
10. Redirect or refresh according to existing project conventions.

Do not duplicate large forms between create and edit pages.

---

# Skill 20: Filter Builder

For filtered lists:

1. Use typed filter state.
2. Reuse existing filter components.
3. Debounce search when appropriate.
4. Sync filters with URL parameters when useful.
5. Reset filters cleanly.
6. Preserve sensible pagination behavior.
7. Avoid excessive filters.
8. Use appropriate controls for master/reference data.
9. Ensure filters have clear labels.
10. Handle loading states for dynamic filter options.

---

# Skill 21: File Upload Builder

For file uploads:

1. Define accepted file types.
2. Define maximum size.
3. Validate before upload.
4. Show selected file.
5. Show upload progress.
6. Show preview when useful.
7. Allow remove/replace.
8. Handle upload failures.
9. Handle retry where useful.
10. Prevent invalid files from being submitted.
11. Follow the backend's actual upload contract.

Do not invent upload endpoints or file restrictions.

---

# Skill 22: Dashboard Widget Builder

Dashboard widgets should:

1. Use shared card components.
2. Have a clear title.
3. Display the primary metric prominently.
4. Show supporting information only when useful.
5. Show trends when meaningful.
6. Provide tooltips/helper text when required.
7. Respect permissions.
8. Respect feature/module availability.
9. Have loading states.
10. Have error states.
11. Have empty states.

Avoid adding charts or metrics merely because dashboard space is available.

---

# Skill 23: Chart Builder

For charts:

1. Use the existing chart library/component system.
2. Use consistent typography.
3. Use semantic colors from the theme.
4. Add labels where necessary.
5. Add tooltips.
6. Add legends only when useful.
7. Handle loading.
8. Handle empty data.
9. Handle errors.
10. Ensure charts remain usable on smaller screens.
11. Avoid unnecessary chart complexity.

A chart should communicate something that is harder to understand from a simple number or table.

---

# Skill 24: Accessibility Builder

For interactive UI:

1. Use semantic HTML.
2. Provide accessible labels.
3. Ensure keyboard navigation.
4. Maintain visible focus states.
5. Use appropriate ARIA only when necessary.
6. Associate form errors with fields.
7. Ensure buttons have meaningful labels.
8. Do not rely solely on color to communicate state.
9. Ensure dialogs and dropdowns are keyboard accessible.
10. Respect reduced-motion preferences where relevant.

Accessibility should be considered during implementation, not added as an afterthought.

---

# Skill 25: State Management Builder

Before adding state:

1. Determine whether local component state is sufficient.
2. Check existing state-management patterns.
3. Avoid global state for local concerns.
4. Avoid duplicating server state in global stores when a data-fetching solution already exists.
5. Reuse existing stores/hooks when appropriate.
6. Keep state ownership clear.
7. Avoid unnecessary synchronization between multiple sources of truth.

Do not introduce Redux/Zustand/another state library unless the project actually needs it.

---

# Skill 26: Data Fetching Builder

For server data:

1. Inspect the project's existing data-fetching approach.
2. Reuse existing hooks/utilities.
3. Handle loading.
4. Handle errors.
5. Handle empty results.
6. Handle stale data where relevant.
7. Invalidate/refetch after mutations where required.
8. Avoid unnecessary duplicate requests.
9. Avoid fetching data that is already available.
10. Keep server state separate from UI state where appropriate.

---

# Skill 27: Permission Guard Builder

For protected routes/features:

1. Check authentication.
2. Check required permissions if the application has RBAC.
3. Check feature/module availability if applicable.
4. Show loading while checks are unresolved.
5. Show forbidden state when access is denied.
6. Show disabled-feature state when appropriate.
7. Prevent unauthorized UI actions.
8. Never rely solely on client-side protection.

---

# Skill 28: Feature Flag / Module Availability Builder

If the application supports feature flags or modules:

1. Inspect the existing feature-flag implementation.
2. Reuse existing checks.
3. Hide unavailable navigation where appropriate.
4. Protect direct routes.
5. Show an appropriate disabled/unavailable state.
6. Prevent unavailable actions.
7. Avoid making unrelated features depend on optional modules.

Do not invent feature flag names.

---

# Skill 29: API Error Mapping

When an API request fails:

1. Inspect the actual error format.
2. Map field-specific validation errors to form fields.
3. Map authorization errors to permission UI.
4. Map not-found errors to not-found states.
5. Map server errors to generic error states.
6. Preserve useful backend messages when safe.
7. Avoid exposing internal implementation details.

Never assume every API failure is a generic `500`.

---

# Skill 30: Testing Builder

When testing infrastructure exists:

1. Add tests for important business behavior.
2. Test form validation.
3. Test critical user interactions.
4. Test permission-sensitive UI.
5. Test loading/error/empty states where meaningful.
6. Test API service behavior where appropriate.
7. Follow existing test conventions.
8. Avoid brittle implementation-detail tests.

Do not add a new testing framework without a strong reason.

---

# Skill 31: Documentation Builder

When implementing a substantial feature, update project documentation when appropriate.

Document:

1. Routes.
2. Components.
3. Services.
4. Types.
5. Permissions.
6. Forms.
7. Tables.
8. Important dependencies.
9. Edge cases.
10. Test coverage.
11. Configuration requirements.

Documentation should describe the actual implementation, not an assumed architecture.

---

# Skill 32: Dependency Builder

Before installing a new package:

1. Check whether the project already has an equivalent dependency.
2. Check whether the requirement can be implemented using existing utilities.
3. Check package compatibility with the project's framework/version.
4. Avoid unnecessary dependencies.
5. Avoid adding large libraries for small requirements.
6. Follow existing package-management conventions.

Every dependency should have a clear reason to exist.

---

# Skill 33: Code Quality Builder

Code should:

1. Be strongly typed.
2. Follow existing naming conventions.
3. Follow existing file organization.
4. Avoid unnecessary duplication.
5. Avoid premature abstraction.
6. Avoid deeply nested components where possible.
7. Avoid unexplained magic values.
8. Keep functions reasonably focused.
9. Keep business rules understandable.
10. Remove dead code created during implementation.

Do not refactor unrelated code merely because it could be written differently.

---

# Skill 34: Hallucination Guard

Before generating implementation code:

1. Inspect existing routes.
2. Inspect existing layouts.
3. Inspect existing components.
4. Inspect existing theme tokens.
5. Inspect existing API services.
6. Inspect existing API/client utilities.
7. Inspect existing type definitions.
8. Inspect existing permission constants.
9. Inspect existing authentication utilities.
10. Inspect existing feature flags.
11. Inspect existing similar pages.

### Never invent without verification:

* API endpoints.
* HTTP methods.
* Request payloads.
* Response formats.
* Permission names.
* Routes.
* Database fields.
* Component APIs.
* Environment variables.
* Feature flags.
* Authentication behavior.

If required information does not exist, identify the missing piece and propose the smallest reasonable addition before depending on it.

---

# Skill 35: Consistency Check

Before considering implementation complete, compare the new feature against at least one existing feature.

Check:

* Folder structure.
* Naming.
* Routing.
* Layout.
* Typography.
* Spacing.
* Buttons.
* Forms.
* Tables.
* Cards.
* Dialogs.
* Toasts.
* Loading states.
* Error states.
* Empty states.
* Permissions.
* API handling.

The new feature should look like it belongs to the existing application.

---

# Skill 36: Regression Check

Before completing frontend work:

1. Run TypeScript/type checking.
2. Run the production build.
3. Run lint if configured.
4. Run relevant tests.
5. Verify the new route loads.
6. Verify navigation still works.
7. Verify existing layout still works.
8. Verify theme still works.
9. Verify permissions still work.
10. Verify forms validate.
11. Verify tables work.
12. Verify loading/error/empty states.
13. Verify responsive behavior.
14. Check browser console for unexpected errors.
15. Check network requests for unexpected failures.
16. Verify no unrelated files were modified.

Do not claim the task is complete if known validation failures remain.

---

# Skill 37: Change Scope Discipline

When implementing a requested feature:

1. Modify only files necessary for the feature.
2. Avoid unrelated refactoring.
3. Avoid changing established architecture without need.
4. Avoid formatting unrelated files.
5. Avoid upgrading dependencies unless required.
6. Avoid changing global styles for a local problem.
7. Keep the diff focused.

If an architectural change is genuinely required, explain why before making broad changes.

---

# Skill 38: Performance Builder

For performance-sensitive UI:

1. Avoid unnecessary client components.
2. Prefer server rendering where appropriate.
3. Avoid unnecessary API calls.
4. Avoid unnecessary re-renders.
5. Use memoization only when it provides value.
6. Lazy-load genuinely heavy components.
7. Optimize large lists where required.
8. Optimize images using the framework's recommended approach.
9. Avoid shipping large dependencies unnecessarily.
10. Measure before making speculative optimizations.

Do not optimize code merely for theoretical performance.

---

# Skill 39: Security-Aware Frontend Builder

Frontend code must not be treated as a security boundary.

1. Never store secrets in client-side code.
2. Never expose private environment variables.
3. Never trust client-provided permissions.
4. Never assume hidden UI means protected data.
5. Avoid rendering unsafe HTML.
6. Sanitize/validate content according to project requirements.
7. Follow secure authentication/session patterns.
8. Do not log sensitive information.
9. Do not expose tokens unnecessarily.
10. Backend authorization remains authoritative.

---

# Skill 40: Final Completion Checklist

A frontend task is complete only when applicable items are satisfied:

* [ ] Existing architecture inspected
* [ ] Existing patterns reused
* [ ] Page implemented
* [ ] Layout consistent
* [ ] Types added/reused
* [ ] API service implemented/reused
* [ ] Validation added where required
* [ ] Loading state added
* [ ] Error state added
* [ ] Empty state added where applicable
* [ ] Not-found state added where applicable
* [ ] Forbidden state added where applicable
* [ ] Authentication handled
* [ ] RBAC handled where applicable
* [ ] Feature/module availability handled where applicable
* [ ] Responsive behavior checked
* [ ] Accessibility considered
* [ ] Theme consistency verified
* [ ] Tests added/updated where appropriate
* [ ] TypeScript check passes
* [ ] Build passes
* [ ] Lint passes where configured
* [ ] No unexpected console errors
* [ ] No unrelated files changed
* [ ] Documentation updated where required

---

# Final Agent Rules

Before writing code:

**Inspect first.**

Before inventing an API:

**Verify first.**

Before creating a component:

**Search for an existing one.**

Before adding state:

**Determine whether existing state is sufficient.**

Before adding a dependency:

**Check whether the project already solves the problem.**

Before adding permissions:

**Inspect the existing permission model.**

Before creating a route:

**Inspect existing route conventions.**

Before finishing:

**Run the project's validation checks.**

When something is missing:

**Do not silently invent it. State the missing dependency and propose the smallest compatible solution.**

The goal is not merely to make the requested page work.

The goal is to make the feature behave as if it was originally designed and implemented as part of the existing application.