import type { APIContext } from "astro";
import { blog as blogConfig, type CategoryId } from "@/config/site";
import { createJsonFeedResponse } from "@/lib/feeds";
import { getPostsByCategory } from "@/lib/posts";

const CATEGORY_IDS = Object.keys(
  blogConfig.categories,
) as CategoryId[];

export async function getStaticPaths() {
  const paths = [];
  for (const category of CATEGORY_IDS) {
    const posts = await getPostsByCategory(category);
    if (posts.length === 0) continue;
    paths.push({ params: { category } });
  }
  return paths;
}

export async function GET(context: APIContext) {
  const category = context.params.category as CategoryId;
  const posts = await getPostsByCategory(category);
  const meta = blogConfig.categories[category];
  return createJsonFeedResponse({
    context,
    posts,
    selfPath: `/blog/${category}/feed.json`,
    title: `${meta.name} Posts`,
    description: meta.description,
  });
}
