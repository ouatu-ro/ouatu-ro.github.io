# AI Maintainer Notes

## Quality gates
- `pnpm astro check` runs the full TypeScript and Astro diagnostics. Install dev dependency `@astrojs/check` (already added) before running locally.
- `pnpm lint:css` executes Stylelint; the configuration rejects vendor-prefixed properties unless explicitly commented and enforces short hex colors.
- Prettier is wired via `pnpm check:format`. Run it before committing large formatting changes.

## Project previews
- All project cards render through `src/components/ProjectPreview.astro`, which now delegates artwork to per-type Astro components in `src/components/project-artworks/`.
- Projects without an explicit `previewType` fall back to the new `titlecard` artwork that prints the full project name and brief description.
- Default preview types live in `src/utils/projects.ts` (`defaultPreviewTypes`). Extend that map or the content collections if you need specific artwork.

## Common pitfalls
- Avoid using `key={...}` on Astro elements rendered on the server; the TypeScript plugin rejects it. Use `data-*` attributes if you need identifiers.
- When adding inline scripts, include the `is:inline` directive or Astro will emit warnings.
- The resume CSS (`src/styles/pages/_resume.css`) is lint-sensitive: keep hex colors short (`#fff` instead of `#ffffff`) and prefer `break-inside` over legacy print properties.

Keep these guardrails in mind so new changes stay lint- and type-clean.
