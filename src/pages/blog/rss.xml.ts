import type { APIContext } from "astro";
import { createRssResponse } from "../../lib/feeds";
import { getAllPosts } from "../../lib/posts";

export async function GET(context: APIContext) {
  const posts = await getAllPosts();
  return createRssResponse({
    context,
    posts,
    selfPath: "/blog/rss.xml",
  });
}
