import rss from "@astrojs/rss";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import fs from "node:fs";
import path from "node:path";
import { slugify } from "../utils/slugify";

// Helper function to read projects data
function readProjectsData() {
  try {
    const projectsFilePath = path.join(
      process.cwd(),
      "public",
      "projects-data.json"
    );
    if (fs.existsSync(projectsFilePath)) {
      const data = JSON.parse(fs.readFileSync(projectsFilePath, "utf8"));
      return data.projects || [];
    }
    console.warn("projects-data.json does not exist, trying manual projects");
  } catch (error) {
    console.warn("Could not read projects-data.json:", error);
  }

  // Try reading manual projects as fallback
  try {
    const manualProjectsPath = path.join(
      process.cwd(),
      "public",
      "manual-projects-data.json"
    );
    if (fs.existsSync(manualProjectsPath)) {
      const data = JSON.parse(fs.readFileSync(manualProjectsPath, "utf8"));
      return data.manualProjects || [];
    }
    console.warn("manual-projects-data.json does not exist");
  } catch (error) {
    console.warn("Could not read manual-projects-data.json:", error);
  }

  return [];
}

export async function GET(context) {
  const projects = readProjectsData().map((project) => ({
    ...project,
    slug: project.slug || slugify(project.name),
  }));

  // Create the RSS feed URL for self-reference
  const rssURL = new URL("projects-rss.xml", context.site).toString();

  return rss({
    title: `${SITE_TITLE} - Projects`,
    description: `Projects and applications created by ${SITE_TITLE}`,
    site: context.site,
    items: projects.map((project) => ({
      title: project.name,
      // Use the real creation date from GitHub if available
      pubDate: project.pubDate ? new Date(project.pubDate) : new Date(),
      description: project.description || "",
      content: `
        <p>${project.description || ""}</p>
        ${
          project.githubUrl
            ? `<p>View the source code on <a href="${project.githubUrl}">GitHub</a>.</p>`
            : ""
        }
        <p><a href="${project.homepage}">Visit the project website</a></p>
        ${
          project.updatedDate
            ? `<p><small>Last updated: ${new Date(
                project.updatedDate
              ).toLocaleDateString()}</small></p>`
            : ""
        }
      `,
      link: `/project/${project.slug}/`,
    })),
    customData: `<language>en-us</language>
    <atom:link href="${rssURL}" rel="self" type="application/rss+xml" />`,
    // Include atom namespace in the XML
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
    },
    stylesheet: "/rss/styles.xsl",
  });
}
