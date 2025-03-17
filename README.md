# My personal website

```sh
pnpm create astro@latest -- --template blog
```

[![Open in StackBlitz](https://developer.stackblitz.com/img/open_in_stackblitz.svg)](https://stackblitz.com/github/withastro/astro/tree/latest/examples/blog)
[![Open with CodeSandbox](https://assets.codesandbox.io/github/button-edit-lime.svg)](https://codesandbox.io/p/sandbox/github/withastro/astro/tree/latest/examples/blog)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/withastro/astro?devcontainer_path=.devcontainer/blog/devcontainer.json)

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

![blog](https://github.com/withastro/astro/assets/2244813/ff10799f-a816-4703-b967-c78997e8323d)

Features:

- ✅ Minimal styling (make it your own!)
- ✅ 100/100 Lighthouse performance
- ✅ SEO-friendly with canonical URLs and OpenGraph data
- ✅ Sitemap support
- ✅ RSS Feed support
- ✅ Markdown & MDX support

## 🚀 Deployment

This site is automatically deployed to GitHub Pages using GitHub Actions. The workflow is configured to:

1. Trigger on pushes to the `master` branch
2. Build the Astro site using PNPM
3. Deploy the built site to GitHub Pages

The deployment configuration is located in:

- `.github/workflows/deploy.yml` - GitHub Actions workflow
- `.github/pages.yml` - GitHub Pages settings

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
├── public/
├── src/
│   ├── components/
│   ├── content/
│   ├── layouts/
│   └── pages/
├── astro.config.mjs
├── README.md
├── package.json
└── tsconfig.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

The `src/content/` directory contains "collections" of related Markdown and MDX documents. Use `getCollection()` to retrieve posts from `src/content/blog/`, and type-check your frontmatter using an optional schema. See [Astro's Content Collections docs](https://docs.astro.build/en/guides/content-collections/) to learn more.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                | Action                                           |
| :--------------------- | :----------------------------------------------- |
| `pnpm install`         | Installs dependencies                            |
| `pnpm dev`             | Starts local dev server at `localhost:4321`      |
| `pnpm build`           | Build your production site to `./dist/`          |
| `pnpm preview`         | Preview your build locally, before deploying     |
| `pnpm astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `pnpm astro -- --help` | Get help using the Astro CLI                     |

## 👀 Want to learn more?

Check out [Astro documentation](https://docs.astro.build) or jump into the [Astro Discord server](https://astro.build/chat).

## Credit

This theme is based off of the lovely [Bear Blog](https://github.com/HermanMartinus/bearblog/).

## Project Data

### Project Publication Dates

The RSS feeds use project publication dates to organize and sort projects. The system handles dates in the following ways:

1. **GitHub Projects**: For projects from GitHub repositories, the system automatically uses:

   - `pubDate`: The repository creation date from GitHub
   - `updatedDate`: The repository's last update date from GitHub

2. **Manual Projects**: For projects added manually in `public/manual-projects-data.json`, you can:
   - Specify a `pubDate` manually in ISO format (e.g., `"2023-01-15T12:00:00Z"`)
   - If not specified, the system will auto-assign a date 2 days before the time of generation

### Adding Manual Projects

To add a project manually, edit the `public/manual-projects-data.json` file with this structure:

```json
{
  "manualProjects": [
    {
      "name": "Project Name",
      "homepage": "https://project-url.com/",
      "description": "Description of the project",
      "githubUrl": null,
      "pubDate": "2023-05-15T10:30:00Z"
    }
  ]
}
```

The `pubDate` field is optional but recommended for proper sorting in the RSS feeds.
