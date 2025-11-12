import { site } from "../config/site";

declare global {
  interface Window {
    dataLayer: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

const GA_ID = site.gaTrackingId;

function initGoogleAnalytics() {
  if (typeof window === "undefined" || !GA_ID) return;

  window.dataLayer = window.dataLayer || [];
  window.gtag =
    window.gtag ||
    function gtag(...args: unknown[]) {
      window.dataLayer.push(args);
    };

  window.gtag("js", new Date());
  window.gtag("config", GA_ID);
}

initGoogleAnalytics();

export {};
