// Function to blink a section when navigated to
function blinkSection(sectionId) {
  const section = document.getElementById(sectionId);
  const originalBgColor = section.style.backgroundColor;
  section.style.backgroundColor = "#454545";
  setTimeout(() => (section.style.backgroundColor = originalBgColor), 500);
}
window.blinkSection = blinkSection;

// Tooltip functionality
function showTooltip(event, text) {
  const tooltip = document.getElementById("tooltip");
  tooltip.innerHTML = text;
  tooltip.style.left = event.pageX + 15 + "px";
  tooltip.style.top = event.pageY + 15 + "px";
  tooltip.style.opacity = 1;
}
function moveTooltip(event) {
  const tooltip = document.getElementById("tooltip");
  tooltip.style.left = event.pageX + 15 + "px";
  tooltip.style.top = event.pageY + 15 + "px";
}
function hideTooltip() {
  const tooltip = document.getElementById("tooltip");
  tooltip.style.opacity = 0;
}

// Projects functionality
const CACHE_KEY = "github_projects";
const CACHE_DURATION = 1000 * 60 * 60; // 1 hour

function createProjectElement(project) {
  const item = document.createElement("li");
  const linkWrapper = document.createElement("div");
  linkWrapper.className = "project-link-wrapper";

  const link = document.createElement("a");
  link.href = project.homepage || project.html_url;
  link.target = "_blank";
  link.textContent = project.name;

  linkWrapper.appendChild(link);

  // Create description element and info button
  if (project.description?.trim()) {
    const infoBtn = document.createElement("button");
    infoBtn.className = "info-btn";
    infoBtn.innerHTML = '<i class="fas fa-info-circle"></i>';
    infoBtn.setAttribute("aria-label", "Show project description");
    linkWrapper.appendChild(infoBtn);

    const descriptionEl = document.createElement("div");
    descriptionEl.className = "project-description";
    descriptionEl.textContent = project.description;

    // Desktop tooltip on info button
    infoBtn.addEventListener("mouseenter", (event) =>
      showTooltip(event, project.description)
    );
    infoBtn.addEventListener("mousemove", moveTooltip);
    infoBtn.addEventListener("mouseleave", hideTooltip);

    // Toggle description on info button click
    infoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const wasActive = descriptionEl.classList.contains("active");
      // Hide all other active descriptions
      document.querySelectorAll(".project-description.active").forEach((el) => {
        el.classList.remove("active");
      });
      if (!wasActive) {
        descriptionEl.classList.add("active");
        // Add a one-time click event to the document to close description
        setTimeout(() => {
          const closeHandler = (event) => {
            if (!item.contains(event.target)) {
              descriptionEl.classList.remove("active");
              document.removeEventListener("click", closeHandler);
            }
          };
          document.addEventListener("click", closeHandler);
        }, 0);
      }
    });

    item.appendChild(linkWrapper);
    item.appendChild(descriptionEl);
  } else {
    item.appendChild(linkWrapper);
  }

  return item;
}

async function fetchAndDisplayProjects() {
  const list = document.getElementById("projects-list");
  if (!list) return;

  // Manual projects
  const manualProjects = [
    {
      name: "Verbalate",
      homepage: "https://verbalate.ai/",
      description:
        "Project for Audio-Visual translation with support for voice cloning and AI lip-sync. For professionals and amateurs alike.",
    },
  ];

  // Display cached data first if available
  const cached = localStorage.getItem(CACHE_KEY);
  if (cached) {
    const { data, timestamp } = JSON.parse(cached);
    if (Date.now() - timestamp < CACHE_DURATION) {
      list.innerHTML = ""; // Clear list before adding cached items
      // Add manual projects
      data.manualProjects.forEach((project) => {
        list.appendChild(createProjectElement(project));
      });
      // Add GitHub projects
      data.githubProjects.forEach((project) => {
        list.appendChild(createProjectElement(project));
      });
    }
  }

  try {
    // Fetch GitHub repositories
    const response = await fetch("https://api.github.com/users/ouatu-ro/repos");
    const repos = await response.json();

    // Filter and process GitHub projects
    const githubProjects = repos.filter(
      (repo) => repo.homepage?.trim() && !repo.fork
    );

    // Clear the list and add fresh data
    list.innerHTML = "";

    // Add manual projects first
    manualProjects.forEach((project) => {
      list.appendChild(createProjectElement(project));
    });

    // Add GitHub projects
    githubProjects.forEach((project) => {
      list.appendChild(createProjectElement(project));
    });

    // Cache the data
    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({
        data: {
          manualProjects,
          githubProjects,
        },
        timestamp: Date.now(),
      })
    );
  } catch (error) {
    console.error("Error fetching repos:", error);
    // If fetch fails and we don't have cached data, show manual projects
    if (!cached) {
      list.innerHTML = "";
      manualProjects.forEach((project) => {
        list.appendChild(createProjectElement(project));
      });
    }
  }
}

// Initialize projects on page load and after navigation
document.addEventListener("DOMContentLoaded", fetchAndDisplayProjects);
document.addEventListener("astro:after-swap", fetchAndDisplayProjects);
