<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" version="1.0" encoding="UTF-8" indent="yes"/>
  <xsl:template match="/">
    <html>
      <head>
        <title><xsl:value-of select="/rss/channel/title"/> Feed</title>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; background: #121212; color: #eee; }
          a { color: #58a6ff; }
          .item { margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #333; }
          .item-description { background: #1e1e1e; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
        </style>
      </head>
      <body>
        <h1><xsl:value-of select="/rss/channel/title"/></h1>
        <p><xsl:value-of select="/rss/channel/description"/></p>
        <p><a href="{/rss/channel/link}">Visit Website</a></p>
        
        <h2>Recent Posts</h2>
        <xsl:for-each select="/rss/channel/item">
          <div class="item">
            <h3><a href="{link}"><xsl:value-of select="title"/></a></h3>
            <p>Published: <xsl:value-of select="pubDate"/></p>
            
            <xsl:if test="description">
              <div class="item-description">
                <xsl:value-of select="description"/>
              </div>
            </xsl:if>
          </div>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
