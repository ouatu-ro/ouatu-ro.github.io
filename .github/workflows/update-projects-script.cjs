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

// Ensure manual-projects-data.json exists with necessary fields
function ensureManualProjectsFileExists() {
  const manualProjectsPath = path.join(
    process.cwd(),
    "public",
    "manual-projects-data.json"
  );

  let manualProjects = [];

  // Try to read existing file
  if (fs.existsSync(manualProjectsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(manualProjectsPath, "utf8"));
      manualProjects = data.manualProjects || [];
      console.log(`Found ${manualProjects.length} existing manual projects`);
    } catch (error) {
      console.warn(
        "Error reading manual-projects-data.json, creating a new one:",
        error
      );
    }
  }

  // Make sure all manual projects have required fields
  manualProjects.forEach((project) => {
    // Add slug if missing
    if (!project.slug) {
      project.slug = slugify(project.name);
    }

    // Add/update pubDate if missing
    if (!project.pubDate) {
      project.pubDate = new Date(
        Date.now() - 2 * 24 * 60 * 60 * 1000
      ).toISOString();
    }

    // Make sure URLs use HTTPS
    if (project.homepage) {
      project.homepage = project.homepage.replace(/^http:\/\//i, "https://");
    }
  });

  // Write the file
  fs.writeFileSync(
    manualProjectsPath,
    JSON.stringify({ manualProjects }, null, 2)
  );

  console.log(
    `Updated manual-projects-data.json with ${manualProjects.length} projects`
  );

  return manualProjects;
}

async function updateProjectsData() {
  // Get GitHub username from command line or use default
  const githubUsername = process.argv[2] || "ouatu-ro";

  console.log(`Fetching repositories for GitHub user: ${githubUsername}`);

  try {
    // Fetch GitHub repositories
    console.log(
      `Fetching repositories from https://api.github.com/users/${githubUsername}/repos`
    );
    const response = await fetch(
      `https://api.github.com/users/${githubUsername}/repos?per_page=100`
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
        homepage: repo.homepage.replace(/^http:\/\//i, "https://"),
        description: repo.description || "",
        githubUrl: repo.html_url,
        slug: slugify(repo.name), // Add slug for reference
        // Add creation date (GitHub's created_at) for the RSS feed
        pubDate: repo.created_at || new Date().toISOString(),
        // Add last update date for additional info
        updatedDate: repo.updated_at || new Date().toISOString(),
      }));

    console.log(
      `Found ${githubProjects.length} GitHub projects with homepages`
    );

    // Ensure manual projects file exists with proper structure
    const manualProjects = ensureManualProjectsFileExists();

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

    // Generate custom project URLs for the sitemap
    const siteUrl = "https://ouatu.ro";
    const projectUrls = allProjects
      .map((project) => {
        return [
          // Project landing page URL
          `${siteUrl}/project/${project.slug}/`,
          // Direct project URL (if it's hosted on the same domain)
          project.homepage.startsWith(siteUrl) ? project.homepage : null,
        ].filter(Boolean); // Remove null values
      })
      .flat();

    // Write to project-urls.json for sitemap generation
    const projectUrlsPath = path.join(publicDir, "project-urls.json");
    fs.writeFileSync(
      projectUrlsPath,
      JSON.stringify({ urls: projectUrls }, null, 2)
    );
    console.log(`Generated ${projectUrls.length} URLs for the sitemap`);

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
