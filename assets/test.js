// Simple test to verify debug script is loaded
setTimeout(() => {
  if (window.albumRaterDebugLoaded) {
    console.log("✅ Debug script is properly loaded and functioning");
  } else {
    console.error("❌ Debug script is NOT loaded properly");
    
    // Attempt to reload it
    const script = document.createElement('script');
    script.src = '/assets/debug.js?force=' + new Date().getTime();
    document.head.appendChild(script);
    console.log("Attempting to reload debug script...");
  }
}, 2000);