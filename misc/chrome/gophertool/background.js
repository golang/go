import {urlForInput} from "./gopher.js";

chrome.omnibox.onInputEntered.addListener(async function (t) {
    const url = urlForInput(t);
    if (url) {
        chrome.tabs.query({"active": true, "currentWindow": true}, async function (tabs) {
            if (tabs.length === 0) {
                await chrome.tabs.create({"url": url});
                return;
            }
            const tab = tabs[0];
            await chrome.tabs.update(tab.id, {"url": url, "selected": true});
        });
    }
});

