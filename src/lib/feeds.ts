import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import MarkdownIt from "markdown-it";
import sanitizeHtml from "sanitize-html";
import { site } from "../config/site";
import type { BlogPost } from "./posts";

const markdown = new MarkdownIt();

const absolute = (site: string, path: string) => new URL(path, site).toString();

function renderContent(post: BlogPost, site: string) {
  const raw = post.body ?? "";
  const rendered = markdown.render(raw);
  return sanitizeHtml(rendered, {
    allowedTags: sanitizeHtml.defaults.allowedTags.concat([
      "img",
      "code",
      "pre",
      "figure",
      "figcaption",
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
          attribs.src = absolute(site, attribs.src);
        }
        return { tagName, attribs };
      },
      a: (tagName, attribs) => {
        if (attribs.href && attribs.href.startsWith("/")) {
          attribs.href = absolute(site, attribs.href);
        }
        return { tagName, attribs };
      },
    },
  });
}

function postLink(post: BlogPost) {
  return `/blog/${post.slug}/`;
}

export function rssItemsFromPosts(posts: BlogPost[], site: string) {
  return posts.map((post) => {
    const mediaImage = post.data.ogImage ?? post.data.heroImage;
    const mediaTag = mediaImage
      ? `<media:content url="${absolute(site, mediaImage)}" medium="image" />`
      : "";
    return {
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      updatedDate: post.data.updatedDate,
      categories: [post.data.category, ...post.normalizedTags.map((tag) => tag.label)],
      link: absolute(site, postLink(post)),
      content: renderContent(post, site),
      customData: mediaTag,
    };
  });
}

export function createRssResponse({
  context,
  posts,
  title = `${site.title} Blog`,
  description = site.description,
  selfPath,
}: {
  context: APIContext;
  posts: BlogPost[];
  title?: string;
  description?: string;
  selfPath: string;
}) {
  const site = String(context.site ?? context.url);
  const feedUrl = absolute(site, selfPath);
  return rss({
    title,
    description,
    site,
    items: rssItemsFromPosts(posts, site),
    customData: `<language>en-us</language>\n<atom:link href="${feedUrl}" rel="self" type="application/rss+xml" />`,
    xmlns: {
      atom: "http://www.w3.org/2005/Atom",
      media: "http://search.yahoo.com/mrss/",
    },
    stylesheet: "/rss/styles.xsl",
  });
}

export function createJsonFeedResponse({
  context,
  posts,
  title = `${site.title} Blog`,
  description = site.description,
  selfPath,
}: {
  context: APIContext;
  posts: BlogPost[];
  title?: string;
  description?: string;
  selfPath: string;
}) {
  const site = String(context.site ?? context.url);
  const feedUrl = absolute(site, selfPath);
  const items = posts.map((post) => {
    const image = post.data.ogImage ?? post.data.heroImage;
    return {
      id: absolute(site, postLink(post)),
      url: absolute(site, postLink(post)),
      title: post.data.title,
      content_html: renderContent(post, site),
      summary: post.data.description,
      image: image ? absolute(site, image) : undefined,
      date_published: post.data.pubDate.toISOString(),
      date_modified: post.data.updatedDate?.toISOString(),
      tags: [post.data.category, ...post.normalizedTags.map((tag) => tag.label)],
    };
  });

  const body = {
    version: "https://jsonfeed.org/version/1.1",
    title,
    description,
    home_page_url: site,
    feed_url: feedUrl,
    items,
  };

  return new Response(JSON.stringify(body, null, 2), {
    status: 200,
    headers: {
      "Content-Type": "application/feed+json; charset=utf-8",
    },
  });
}
