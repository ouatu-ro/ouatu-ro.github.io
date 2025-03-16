import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import sanitizeHtml from "sanitize-html";
import MarkdownIt from "markdown-it";

const parser = new MarkdownIt();

export async function GET(context) {
  const posts = await getCollection("blog");

  // Log the first post structure for debugging (will show in build logs)
  if (posts.length > 0) {
    console.log("RSS Debug - First post structure:", {
      id: posts[0].id,
      slug: posts[0].slug,
      collection: posts[0].collection,
      data: {
        title: posts[0].data.title,
        pubDate: posts[0].data.pubDate,
      },
      bodyPreview: posts[0].body
        ? posts[0].body.slice(0, 100) + "..."
        : "No body",
    });
  }

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: post.data.description,
      // Process Markdown to HTML and sanitize it
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
      }),
      // Add image enclosure if available
      customData: post.data.heroImage
        ? `<enclosure url="${new URL(
            post.data.heroImage,
            context.site
          )}" type="image/jpeg" />`
        : "",
      link: `/blog/${post.id}/`,
    })),
    // Add XML language declaration
    customData: `<language>en-us</language>`,
    // Add a stylesheet for better feed display in browsers
    stylesheet: "/rss/styles.xsl",
  });
}
