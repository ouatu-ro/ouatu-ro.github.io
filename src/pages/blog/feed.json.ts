import type { APIContext } from "astro";
import { createJsonFeedResponse } from "../../lib/feeds";
import { getAllPosts } from "../../lib/posts";

export async function GET(context: APIContext) {
  const posts = await getAllPosts();
  return createJsonFeedResponse({
    context,
    posts,
    selfPath: "/blog/feed.json",
  });
}
