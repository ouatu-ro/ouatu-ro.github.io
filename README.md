# My personal website

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
