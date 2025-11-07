import type { APIContext } from "astro";
import { CATEGORIES, type CategoryId } from "@/config/categories";
import { createRssResponse } from "@/lib/feeds";
import { getPostsByCategory } from "@/lib/posts";

export async function getStaticPaths() {
  const categories = Object.keys(CATEGORIES) as CategoryId[];
  const paths = [];
  for (const category of categories) {
    const posts = await getPostsByCategory(category);
    if (posts.length === 0) continue;
    paths.push({ params: { category } });
  }
  return paths;
}

export async function GET(context: APIContext) {
  const category = context.params.category as CategoryId;
  const posts = await getPostsByCategory(category);
  const meta = CATEGORIES[category];
  return createRssResponse({
    context,
    posts,
    selfPath: `/blog/category/${category}/rss.xml`,
    title: `${meta.name} Posts`,
    description: meta.description,
  });
}
