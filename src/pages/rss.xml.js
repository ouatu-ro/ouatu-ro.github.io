import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import sanitizeHtml from "sanitize-html";
import MarkdownIt from "markdown-it";
import fs from "node:fs";
import path from "node:path";
import { slugify } from "../utils/slugify";

const parser = new MarkdownIt();

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
  // Get blog posts
  const posts = await getCollection("blog");

  // Get projects
  const projects = readProjectsData().map((project) => ({
    ...project,
    slug: project.slug || slugify(project.name),
  }));

  // Create RSS feed items from blog posts
  const blogItems = posts.map((post) => {
    const description =
      post.data.description || post.data.summary || post.data.excerpt || "";

    return {
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: description,
      content: sanitizeHtml(parser.render(post.body || ""), {
        allowedTags: sanitizeHtml.defaults.allowedTags.concat([
          "img",
          "code",
          "pre",
          "h1",
          "h2",
          "h3",
          "h4",
          "h5",
          "h6",
        ]),
        allowedAttributes: {
          ...sanitizeHtml.defaults.allowedAttributes,
          img: ["src", "alt", "title", "width", "height", "loading"],
          a: ["href", "name", "target", "rel", "title"],
          code: ["class"],
          pre: ["class"],
        },
        transformTags: {
          img: (tagName, attribs) => {
            if (attribs.src && attribs.src.startsWith("/")) {
              attribs.src = new URL(attribs.src, context.site).toString();
            }
            return { tagName, attribs };
          },
          a: (tagName, attribs) => {
            if (attribs.href && attribs.href.startsWith("/")) {
              attribs.href = new URL(attribs.href, context.site).toString();
            }
            return { tagName, attribs };
          },
        },
      }),
      customData: post.data.heroImage
        ? `<enclosure url="${new URL(
            post.data.heroImage,
            context.site
          )}" type="image/jpeg" />`
        : "",
      link: `/blog/${post.id}/`,
      categories: ["blog"],
    };
  });

  // Create RSS feed items from projects
  const projectItems = projects.map((project) => ({
    title: `Project: ${project.name}`,
    // Use the real creation date from GitHub if available, otherwise use a fallback date
    pubDate: project.pubDate
      ? new Date(project.pubDate)
      : new Date(Date.now() - 24 * 60 * 60 * 1000),
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
    categories: ["project"],
  }));

  // Combine and sort all items by date (newest first)
  const allItems = [...blogItems, ...projectItems].sort(
    (a, b) => new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime()
  );

  // Create the RSS feed URL for self-reference
  const rssURL = new URL("rss.xml", context.site).toString();

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: allItems,
    customData: `<language>en-us</language>
    <atom:link href="${rssURL}" rel="self" type="application/rss+xml" />`,
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
    },
    stylesheet: "/rss/styles.xsl",
  });
}
