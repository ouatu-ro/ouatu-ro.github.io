import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import sanitizeHtml from "sanitize-html";
import MarkdownIt from "markdown-it";

const parser = new MarkdownIt();

export async function GET(context) {
  const posts = (await getCollection("blog")).sort((a, b) => {
    return (
      new Date(b.data.pubDate).getTime() - new Date(a.data.pubDate).getTime()
    );
  });

  // Create the RSS feed URL for self-reference
  const rssURL = new URL("blog-rss.xml", context.site).toString();

  return rss({
    title: `${SITE_TITLE} - Blog Posts`,
    description: `Blog posts from ${SITE_TITLE}`,
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
          // Transform relative URLs to absolute URLs
          transformTags: {
            img: (tagName, attribs) => {
              // Convert relative image URLs to absolute
              if (attribs.src && attribs.src.startsWith("/")) {
                attribs.src = new URL(attribs.src, context.site).toString();
              }
              return { tagName, attribs };
            },
            a: (tagName, attribs) => {
              // Convert relative link URLs to absolute
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
    }),
    customData: `<language>en-us</language>
    <atom:link href="${rssURL}" rel="self" type="application/rss+xml" />`,
    // Include atom namespace in the XML
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
    },
    stylesheet: "/rss/styles.xsl",
  });
}
