chrome.omnibox.onInputEntered.addListener(function(t) {
  var url = urlForInput(t);
  if (url) {
    chrome.tabs.query({ "active": true, "currentWindow": true }, function(tab) {
      if (!tab) return;
      chrome.tabs.update(tab.id, { "url": url, "selected": true });
    });
  }
});
