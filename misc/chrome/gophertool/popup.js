function openURL(url) {
  chrome.tabs.create({ "url": url })
}

function addLinks() {
  const links = document.getElementsByTagName("a");
  for (let i = 0; i < links.length; i++) {
    const url = links[i].getAttribute("url");
    if (url)
      links[i].addEventListener("click", function () {
        openURL(this.getAttribute("url"));
      });
  }
}

window.addEventListener("load", function () {
  addLinks();
  console.log("hacking gopher pop-up loaded.");
  document.getElementById("inputbox").focus();
});

window.addEventListener("submit", function () {
  console.log("submitting form");
  const box = document.getElementById("inputbox");
  box.focus();

  const t = box.value;
  if (t === "") {
    return false;
  }

  const success = function (url) {
    console.log("matched " + t + " to: " + url)
    box.value = "";
    openURL(url);
    return false;  // cancel form submission
  };

  const url = urlForInput(t);
  if (url) {
    return success(url);
  }

  console.log("no match for text: " + t)
  return false;
});
