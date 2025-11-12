declare global {
  interface DocumentEventMap {
    "astro:page-load": Event;
    "astro:after-swap": Event;
  }

  interface Window {
    __projectPreviewObserver?: IntersectionObserver;
  }
}

let cleanupBurgerMenu: (() => void) | null = null;

function setupBurgerMenu() {
  // Remove previous handlers before wiring up the current header instance.
  cleanupBurgerMenu?.();
  cleanupBurgerMenu = null;

  const burgerMenu = document.querySelector<HTMLElement>(".burger-menu");
  const navLinks = document.querySelector<HTMLElement>(".nav-links");

  if (!burgerMenu || !navLinks) return;

  const toggleMenu = () => {
    burgerMenu.classList.toggle("active");
    navLinks.classList.toggle("active");
  };

  const closeOnClickOutside = (event: MouseEvent) => {
    const target = event.target as HTMLElement | null;
    if (
      !target ||
      target.closest(".burger-menu") ||
      target.closest(".nav-links") ||
      !navLinks.classList.contains("active")
    ) {
      return;
    }

    burgerMenu.classList.remove("active");
    navLinks.classList.remove("active");
  };

  burgerMenu.addEventListener("click", toggleMenu);
  document.addEventListener("click", closeOnClickOutside);

  cleanupBurgerMenu = () => {
    burgerMenu.removeEventListener("click", toggleMenu);
    document.removeEventListener("click", closeOnClickOutside);
  };
}

function addCopyButtonsToCodeBlocks() {
  const codeBlocks = document.querySelectorAll<HTMLElement>(
    'pre[data-language]:not([data-language="plaintext"])',
  );

  codeBlocks.forEach((block) => {
    if (block.querySelector(".copy-button")) return;

    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';

    copyButton.addEventListener("click", () => {
      const code = block.querySelector("code");
      let textToCopy = "";

      if (code) {
        const lines = Array.from(code.querySelectorAll(".line"));
        textToCopy = lines.map((line) => line.textContent).join("\n");
      } else {
        textToCopy = block.innerText;
      }

      navigator.clipboard
        .writeText(textToCopy)
        .then(() => {
          copyButton.classList.add("copied");
          copyButton.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';

          setTimeout(() => {
            copyButton.classList.remove("copied");
            copyButton.innerHTML =
              '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';
          }, 2000);
        })
        .catch((error) => {
          console.error("Failed to copy code: ", error);
          copyButton.textContent = "Error!";
        });
    });

    block.appendChild(copyButton);
  });
}

function initProjectPreviewVideos() {
  const root = window;

  if (!root.__projectPreviewObserver) {
    root.__projectPreviewObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;

          const video = entry.target as HTMLVideoElement;
          const dataSrc = video.getAttribute("data-src");

          if (dataSrc && !video.getAttribute("src")) {
            video.setAttribute("src", dataSrc);
            video.load();
            video.play().catch(() => {});
          }

          observer.unobserve(video);
        });
      },
      { rootMargin: "600px 0px", threshold: 0.01 },
    );
  }

  const candidates = document.querySelectorAll<HTMLVideoElement>(
    ".project-card-media video[data-src]",
  );

  const viewportH = window.innerHeight || document.documentElement.clientHeight;

  candidates.forEach((video) => {
    if (video.getAttribute("data-observed")) return;
    video.setAttribute("data-observed", "true");

    const setAndPlay = () => {
      const dataSrc = video.getAttribute("data-src");
      if (dataSrc && !video.getAttribute("src")) {
        video.setAttribute("src", dataSrc);
        video.load();
        video.play().catch(() => {});
      }
    };

    // Eagerly load if already near viewport (approx. 600px margin)
    const rect = video.getBoundingClientRect();
    const nearViewport = rect.top <= viewportH + 600 && rect.bottom >= -600;
    if (nearViewport) {
      setAndPlay();
      return;
    }

    root.__projectPreviewObserver?.observe(video);

    // Safety timer: try again after styles/layout settle
    setTimeout(() => {
      if (!video.getAttribute("src")) setAndPlay();
    }, 250);
  });
}

function setupProjectInfoButtons() {
  const infoButtons = document.querySelectorAll<HTMLElement>(".info-btn");
  if (!infoButtons.length) return;

  let tooltip = document.getElementById("tooltip");

  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "tooltip";
    document.body.appendChild(tooltip);
  }

  infoButtons.forEach((button) => {
    const description = button
      .closest(".project-link-wrapper")
      ?.nextElementSibling as HTMLElement | null;

    if (!description || !description.classList.contains("project-description")) {
      return;
    }

    button.addEventListener("click", () => {
      description.classList.toggle("active");
    });

    if (window.innerWidth > 768) {
      button.addEventListener("mouseenter", () => {
        const descText = description.textContent || "";
        tooltip!.textContent = descText;

        const buttonRect = button.getBoundingClientRect();
        const scrollLeft =
          window.pageXOffset || document.documentElement.scrollLeft;
        const scrollTop =
          window.pageYOffset || document.documentElement.scrollTop;

        tooltip!.style.opacity = "0";
        tooltip!.style.visibility = "visible";
        tooltip!.getBoundingClientRect();

        const tooltipHeight = tooltip!.offsetHeight;
        const tooltipWidth = tooltip!.offsetWidth;
        const absoluteTop = buttonRect.top + scrollTop;
        const absoluteLeft = buttonRect.left + scrollLeft;
        const absoluteRight = buttonRect.right + scrollLeft;
        const viewportWidth = window.innerWidth;

        let leftPos: number;

        if (absoluteRight + 10 + tooltipWidth > scrollLeft + viewportWidth) {
          leftPos = absoluteLeft - tooltipWidth - 10;
          tooltip!.classList.add("tooltip-left");
        } else {
          leftPos = absoluteRight + 10;
          tooltip!.classList.remove("tooltip-left");
        }

        let topPos = absoluteTop + buttonRect.height / 2 - tooltipHeight / 2;
        const viewportHeight = window.innerHeight;

        if (topPos + tooltipHeight > scrollTop + viewportHeight) {
          topPos = scrollTop + viewportHeight - tooltipHeight - 10;
        }

        if (topPos < scrollTop) {
          topPos = scrollTop + 10;
        }

        tooltip!.style.left = `${leftPos}px`;
        tooltip!.style.top = `${topPos}px`;
        tooltip!.style.opacity = "1";
      });

      button.addEventListener("mouseleave", () => {
        tooltip!.style.opacity = "0";
        tooltip!.style.visibility = "hidden";
      });
    }
  });

  document.addEventListener("click", (event) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest(".info-btn") || target?.closest(".project-description")) {
      return;
    }

    document
      .querySelectorAll<HTMLElement>(".project-description.active")
      .forEach((desc) => desc.classList.remove("active"));
  });
}

function blinkSection(sectionId: string) {
  const section = document.getElementById(sectionId);
  if (!section) return;

  section.style.backgroundColor = "#454545";
  setTimeout(() => {
    section.style.backgroundColor = "#121212";
  }, 500);
}

function scrollToSection(hash: string) {
  const sectionId = hash.replace(/^#/, "");
  if (!sectionId) return;

  const section = document.getElementById(sectionId);
  if (!section) return;

  section.scrollIntoView();
  blinkSection(sectionId);
}

function handleInitialHash() {
  const hash = window.location.hash;
  if (!hash) return;

  setTimeout(() => scrollToSection(hash), 100);
}

function closeMobileMenu() {
  const burgerMenu = document.querySelector<HTMLElement>(".burger-menu");
  const navLinks = document.querySelector<HTMLElement>(".nav-links");

  if (!burgerMenu || !navLinks) return;

  burgerMenu.classList.remove("active");
  navLinks.classList.remove("active");
}

function setupNavigation() {
  const header = document.querySelector("header");
  if (!header || header.hasAttribute("data-nav-initialized")) return;

  header.setAttribute("data-nav-initialized", "true");

  document.querySelectorAll<HTMLAnchorElement>(".nav-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      const section = link.getAttribute("data-section");
      if (!section) return;

      const targetSection = document.getElementById(section);
      if (!targetSection) return;

      event.preventDefault();
      targetSection.scrollIntoView();
      blinkSection(section);
      history.pushState(null, "", `/#${section}`);
      closeMobileMenu();
    });
  });

  document.querySelector<HTMLAnchorElement>(".home-link")?.addEventListener(
    "click",
    (event) => {
      if (window.location.pathname === "/") {
        event.preventDefault();
        window.scrollTo({ top: 0 });
      }
    },
  );
}

function onDomReady() {
  setupBurgerMenu();
  addCopyButtonsToCodeBlocks();
  setupProjectInfoButtons();
  setupNavigation();
  initProjectPreviewVideos();
  handleInitialHash();
}

document.addEventListener("DOMContentLoaded", onDomReady);
document.addEventListener("astro:page-load", () => {
  initProjectPreviewVideos();
  handleInitialHash();
});
document.addEventListener("astro:after-swap", () => {
  initProjectPreviewVideos();
  handleInitialHash();
});

// If the script loads after DOM is ready, initialize immediately
if (document.readyState !== "loading") {
  onDomReady();
  // Run a second pass after a tick to catch late layout/images
  setTimeout(initProjectPreviewVideos, 0);
  setTimeout(initProjectPreviewVideos, 250);
  setTimeout(initProjectPreviewVideos, 1000);
}

// Catch late DOM insertions (e.g., after client-side routing/content swaps)
const mo = new MutationObserver(() => initProjectPreviewVideos());
mo.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("popstate", handleInitialHash);

export {};
