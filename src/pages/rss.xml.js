import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import sanitizeHtml from "sanitize-html";
import MarkdownIt from "markdown-it";

const parser = new MarkdownIt();

export async function GET(context) {
  const posts = await getCollection("blog");

  //   // Debug: Log the first post's complete data structure
  //   if (posts.length > 0) {
  //     console.log("RSS Debug - First post data:", {
  //       id: posts[0].id,
  //       data: posts[0].data,
  //       // Show what fields are available in the data object
  //       dataKeys: Object.keys(posts[0].data),
  //     });
  //   }

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => {
      // Make sure to include description - try different field names if needed
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
        }),
        customData: post.data.heroImage
          ? `<enclosure url="${new URL(
              post.data.heroImage,
              context.site
            )}" type="image/jpeg" />`
          : "",
        link: `/blog/${post.id}/`,
      };
    }),
    customData: `<language>en-us</language>`,
    stylesheet: "/rss/styles.xsl",
  });
}
