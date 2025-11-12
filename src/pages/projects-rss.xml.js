import rss from "@astrojs/rss";
import { site } from "../config/site";
import { getAllProjects } from "../lib/projects";

export async function GET(context) {
  const projects = (await getAllProjects()).sort((a, b) => {
    const dateA = new Date(a.pubDate || a.updatedDate || 0);
    const dateB = new Date(b.pubDate || b.updatedDate || 0);
    return dateB.getTime() - dateA.getTime();
  });
  const rssURL = new URL("projects-rss.xml", context.site).toString();

  return rss({
    title: `${site.title} - Projects`,
    description: `Projects and applications created by ${site.title}`,
    site: context.site,
    items: projects.map((project) => ({
      title: project.name,
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
                project.updatedDate,
              ).toLocaleDateString()}</small></p>`
            : ""
        }
      `,
      link: `/project/${project.slug}/`,
    })),
    customData: `<language>en-us</language>
    <atom:link href="${rssURL}" rel="self" type="application/rss+xml" />`,
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
    },
    stylesheet: "/rss/styles.xsl",
  });
}
