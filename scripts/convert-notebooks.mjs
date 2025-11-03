import { watch } from "fs";
import { execSync } from "child_process";
import path from "path";
import { existsSync, promises as fs } from "fs";
import process from "process";

// Configuration
const NOTEBOOK_DIR = "src/content/notebooks";
const LABS_DIR = "src/content/labs";
const JUPYTER_PATH = ".venv/bin/jupyter";

async function ensureLabsDir() {
  if (!existsSync(LABS_DIR)) {
    await fs.mkdir(LABS_DIR, { recursive: true });
  }
}

// Check if jupyter exists
if (!existsSync(JUPYTER_PATH)) {
  console.error(`❌ Jupyter not found at ${JUPYTER_PATH}`);
  console.error("Please install it with: uv pip install jupyter");
  process.exit(1);
}

// Convert a notebook to markdown
async function convertNotebook(notebookPath) {
  try {
    // Skip non-notebook files
    if (!notebookPath.endsWith(".ipynb")) {
      return;
    }

    const basename = path.basename(notebookPath, ".ipynb");
    console.log(`🔄 Converting ${basename}...`);

    await ensureLabsDir();

    // Run jupyter nbconvert synchronously with additional options for math
    execSync(
      `"${JUPYTER_PATH}" nbconvert --to markdown "${notebookPath}" --output-dir "${LABS_DIR}" --output "${basename}.md"`,
      { stdio: "inherit" }
    );

    // Add frontmatter and fix math expressions if needed
    const mdPath = path.join(LABS_DIR, `${basename}.md`);
    let content = await fs.readFile(mdPath, "utf8");

    // Fix potentially broken math delimiters
    // Replace escaped $ with actual $
    // content = content.replace(/\\$\\$/g, "$$").replace(/\\\$/g, "$");

    // Ensure display math is properly formatted (double dollar signs at start and end of line)
    // content = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, p1) => {
    //   return `\n$$\n${p1}\n$$\n`;
    // });

    // Add frontmatter if needed
    if (!content.includes("---")) {
      const frontmatter = `---
title: "${basename}"
pubDate: ${new Date().toISOString().split("T")[0]}
description: ""
math: true
---

`;
      content = frontmatter + content;
    } else if (!content.includes("math: true")) {
      // If frontmatter exists, ensure math: true is set
      content = content.replace(/^---\n/, "---\nmath: true\n");
    }

    await fs.writeFile(mdPath, content, "utf8");
    console.log(`✅ Converted ${basename} successfully!`);
  } catch (error) {
    console.error(`❌ Error converting notebook:`, error.message);
  }
}

// Initial conversion of all notebooks
async function initialConversion() {
  try {
    await ensureLabsDir();
    console.log("🔍 Finding existing notebooks...");
    const files = await fs.readdir(NOTEBOOK_DIR);
    const notebooks = files.filter((file) => file.endsWith(".ipynb"));

    console.log(`Found ${notebooks.length} notebooks`);

    for (const notebook of notebooks) {
      await convertNotebook(path.join(NOTEBOOK_DIR, notebook));
    }
  } catch (error) {
    console.error("❌ Error during initial conversion:", error);
  }
}

// Process all notebooks on startup
initialConversion();

// Use native fs.watch
console.log(`🚀 Watching for changes in ${NOTEBOOK_DIR}...`);

// Track last modification time to prevent duplicate events
const lastModified = new Map();

// Watch the directory for changes
watch(NOTEBOOK_DIR, { recursive: true }, async (eventType, filename) => {
  if (filename && filename.endsWith(".ipynb")) {
    const fullPath = path.join(NOTEBOOK_DIR, filename);

    try {
      // Check if file exists and get stats
      const stats = await fs.stat(fullPath);
      const mtime = stats.mtime.getTime();

      // Only process if file is newer than last time
      if (!lastModified.has(filename) || lastModified.get(filename) < mtime) {
        console.log(`📝 Notebook changed: ${filename}`);
        lastModified.set(filename, mtime);
        await convertNotebook(fullPath);
      }
    } catch (error) {
      // File might have been deleted or moved
      if (error.code === "ENOENT") {
        console.log(`🗑️ Notebook removed: ${filename}`);
        lastModified.delete(filename);
      } else {
        console.error(`❌ Error processing ${filename}:`, error);
      }
    }
  }
});

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.log("👋 Stopping watcher...");
  process.exit(0);
});
