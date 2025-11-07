import type { APIContext } from "astro";
import { createJsonFeedResponse } from "@/lib/feeds";
import { getAllTags, getPostsByTag } from "@/lib/posts";

export async function getStaticPaths() {
  const tags = await getAllTags();
  return tags.map((tag) => ({ params: { tag: tag.slug }, props: { tagLabel: tag.label } }));
}

export async function GET(context: APIContext) {
  const tagSlug = context.params.tag as string;
  const posts = await getPostsByTag(tagSlug);
  const label = context.props?.tagLabel ?? tagSlug;
  return createJsonFeedResponse({
    context,
    posts,
    selfPath: `/blog/tag/${tagSlug}/feed.json`,
    title: `Posts tagged ${label}`,
    description: `Posts filtered by ${label}`,
  });
}
