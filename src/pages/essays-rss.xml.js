import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import sanitizeHtml from "sanitize-html";
import MarkdownIt from "markdown-it";

const parser = new MarkdownIt();

export async function GET(context) {
  const siteOrigin = context.site ?? context.url;
  const essays = (await getCollection("essays")).sort((a, b) => {
    const first = a.data.pubDate ?? a.data.updatedDate ?? new Date(0);
    const second = b.data.pubDate ?? b.data.updatedDate ?? new Date(0);
    return second.getTime() - first.getTime();
  });

  const rssURL = new URL("essays-rss.xml", siteOrigin).toString();

  return rss({
    title: `${SITE_TITLE} - Essays`,
    description: `Essays and long-form writing from ${SITE_TITLE}`,
    site: siteOrigin,
    items: essays.map((essay) => {
      const description =
        essay.data.description ||
        essay.data.summary ||
        essay.data.excerpt ||
        SITE_DESCRIPTION;

      return {
        title: essay.data.title,
        pubDate:
          essay.data.pubDate ?? essay.data.updatedDate ?? new Date(),
        description,
        content: sanitizeHtml(parser.render(essay.body || ""), {
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
                attribs.src = new URL(attribs.src, siteOrigin).toString();
              }
              return { tagName, attribs };
            },
            a: (tagName, attribs) => {
              if (attribs.href && attribs.href.startsWith("/")) {
                attribs.href = new URL(attribs.href, siteOrigin).toString();
              }
              return { tagName, attribs };
            },
          },
        }),
        link: `/essays/${essay.id}/`,
        categories: ["essays"],
      };
    }),
    customData: `<language>en-us</language>
    <atom:link href="${rssURL}" rel="self" type="application/rss+xml" />`,
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
    },
    stylesheet: "/rss/styles.xsl",
  });
}
