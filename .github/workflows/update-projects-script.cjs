#!/usr/bin/env node

/**
 * Update Projects Data
 *
 * This script fetches GitHub repositories and combines them with manual projects
 * to create a comprehensive projects-data.json file.
 *
 * Usage:
 *   node .github/workflows/update-projects-script.cjs [github-username]
 *
 * If no GitHub username is provided, it will use "ouatu-ro" as the default.
 */

const fs = require("fs");
const fetch = require("node-fetch");
const path = require("path");

// Include the same slugify function used elsewhere
function slugify(text) {
  return text
    .toString()
    .toLowerCase()
    .replace(/\s+/g, "-") // Replace spaces with -
    .replace(/[^\w\-]+/g, "") // Remove all non-word chars
    .replace(/\-\-+/g, "-") // Replace multiple - with single -
    .trim(); // Trim - from start and end
}

// Also copy the slugify function to the public script file to ensure consistency
function ensureSlugifyScriptExists() {
  // Make sure the scripts directory exists
  const scriptsDir = path.join(process.cwd(), "public", "scripts");
  if (!fs.existsSync(scriptsDir)) {
    fs.mkdirSync(scriptsDir, { recursive: true });
  }

  fs.writeFileSync(
    path.join(scriptsDir, "slugify.js"),
    `// Function to convert project name to slug
function slugify(text) {
  return text
    .toString()
    .toLowerCase()
    .replace(/\\s+/g, '-')       // Replace spaces with -
    .replace(/[^\\w\\-]+/g, '')   // Remove all non-word chars
    .replace(/\\-\\-+/g, '-')     // Replace multiple - with single -
    .trim();                    // Trim - from start and end
}

// Make it available for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { slugify };
}
`
  );
}

async function readManualProjects() {
  try {
    // Read manual projects from public/manual-projects-data.json
    const manualProjectsPath = path.join(
      process.cwd(),
      "public",
      "manual-projects-data.json"
    );
    if (fs.existsSync(manualProjectsPath)) {
      const data = JSON.parse(fs.readFileSync(manualProjectsPath, "utf8"));
      console.log(
        `Successfully loaded ${
          data.manualProjects?.length || 0
        } manual projects from manual-projects-data.json`
      );
      return data.manualProjects || [];
    }
  } catch (error) {
    console.warn("Error reading manual-projects-data.json:", error);
  }

  // Fallback to default manual project if file doesn't exist or can't be read
  console.log("Using default manual project as fallback");
  return [
    {
      name: "Verbalate",
      homepage: "https://verbalate.ai/",
      description:
        "Project for Audio-Visual translation with support for voice cloning and AI lip-sync. For professionals and amateurs alike.",
      githubUrl: null,
    },
  ];
}

async function updateProjectsData() {
  // Get GitHub username from command line or use default
  const githubUsername = process.argv[2] || "ouatu-ro";

  console.log(`Fetching repositories for GitHub user: ${githubUsername}`);

  try {
    // Ensure slugify.js exists
    ensureSlugifyScriptExists();

    // Get manual projects from JSON file
    const manualProjects = await readManualProjects();

    // Fetch GitHub repositories
    console.log(
      `Fetching repositories from https://api.github.com/users/${githubUsername}/repos`
    );
    const response = await fetch(
      `https://api.github.com/users/${githubUsername}/repos`
    );

    if (!response.ok) {
      throw new Error(
        `GitHub API responded with ${response.status}: ${response.statusText}`
      );
    }

    const repos = await response.json();
    console.log(`Found ${repos.length} total repositories`);

    // Filter and process GitHub projects
    const githubProjects = repos
      .filter((repo) => repo.homepage && repo.homepage.trim() && !repo.fork)
      .map((repo) => ({
        name: repo.name,
        homepage: repo.homepage,
        description: repo.description || "",
        githubUrl: repo.html_url,
        slug: slugify(repo.name), // Add slug for reference
      }));

    console.log(
      `Found ${githubProjects.length} GitHub projects with homepages`
    );

    // Add slugs to manual projects too
    manualProjects.forEach((project) => {
      project.slug = slugify(project.name);
    });

    // Combine all projects
    const allProjects = [...manualProjects, ...githubProjects];

    // Ensure public directory exists
    const publicDir = path.join(process.cwd(), "public");
    if (!fs.existsSync(publicDir)) {
      fs.mkdirSync(publicDir, { recursive: true });
    }

    // Write to projects-data.json
    const projectsDataPath = path.join(publicDir, "projects-data.json");
    fs.writeFileSync(
      projectsDataPath,
      JSON.stringify({ projects: allProjects }, null, 2)
    );

    console.log(
      `Successfully updated ${projectsDataPath} with ${allProjects.length} projects`
    );

    // List all projects added
    console.log("\nProjects added:");
    allProjects.forEach((project) => {
      console.log(`- ${project.name} (${project.slug}): ${project.homepage}`);
    });
  } catch (error) {
    console.error("Error updating projects data:", error);
    process.exit(1);
  }
}

// Run the script
updateProjectsData();
