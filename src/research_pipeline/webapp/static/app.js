/* Fragment polling, in place of a client-side framework.
 *
 * The server renders every fragment; this only asks for a fresh one on an
 * interval and swaps it in. An element opts in with data-poll-url, so a
 * fragment stops polling simply by being rendered without that attribute —
 * which is how a finished run's panes go quiet.
 *
 * A failed fetch reschedules instead of giving up: the intended deployment is
 * an SSH tunnel to a compute node, and tunnels drop.
 */
(function () {
  "use strict";

  function schedule(el) {
    var url = el.dataset.pollUrl;
    if (!url) return;
    var delay = parseInt(el.dataset.pollInterval || "2000", 10);

    setTimeout(function () {
      if (!el.isConnected) return; // already replaced by a newer fragment
      fetch(url, { headers: { "X-Requested-With": "fetch" } })
        .then(function (response) {
          if (!response.ok) throw new Error(response.status);
          return response.text();
        })
        .then(function (html) {
          swap(el, html);
        })
        .catch(function () {
          schedule(el); // transient: try again on the next tick
        });
    }, delay);
  }

  function swap(el, html) {
    var holder = document.createElement("div");
    holder.innerHTML = html.trim();
    var next = holder.firstElementChild;
    if (!next) return;

    var scrollers = capture(el);
    el.replaceWith(next);
    restore(next, scrollers);
    attach(next);
  }

  /* Preserve the log pane's scroll position across a swap, so reading
   * something mid-run isn't interrupted every two seconds. */
  function capture(root) {
    var state = [];
    root.querySelectorAll("[data-autoscroll]").forEach(function (pane, index) {
      var atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
      state[index] = { atBottom: atBottom, top: pane.scrollTop };
    });
    return state;
  }

  function restore(root, state) {
    root.querySelectorAll("[data-autoscroll]").forEach(function (pane, index) {
      var previous = state[index];
      pane.scrollTop = !previous || previous.atBottom ? pane.scrollHeight : previous.top;
    });
  }

  function attach(root) {
    if (root.matches && root.matches("[data-poll-url]")) schedule(root);
    if (root.querySelectorAll) root.querySelectorAll("[data-poll-url]").forEach(schedule);
    if (root.querySelectorAll) {
      root.querySelectorAll("[data-autoscroll]").forEach(function (pane) {
        pane.scrollTop = pane.scrollHeight;
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    attach(document);
  });
})();
