export async function GET(context) {
  const target = new URL("labs-rss.xml", context.site ?? context.url);
  return Response.redirect(target, 308);
}
