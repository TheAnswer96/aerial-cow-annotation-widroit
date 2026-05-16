/**
 * Shared utility: SSE-based task progress streaming.
 *
 * opts = {
 *   fillId,   // progress bar fill div id
 *   textId,   // small text below bar id
 *   pctId,    // percentage label id
 *   msgId,    // top message label id
 *   onStep,   // fn(data) called on each progress event
 *   onDone,   // fn(data) called when done event received
 * }
 */
function streamProgress(taskId, opts) {
  const es = new EventSource('/progress/' + taskId);

  es.onmessage = function (e) {
    const data = JSON.parse(e.data);

    if (data.msg && opts.msgId) {
      document.getElementById(opts.msgId).textContent = data.msg;
    }

    if (data.pct !== undefined) {
      if (opts.fillId) document.getElementById(opts.fillId).style.width = data.pct + '%';
      if (opts.pctId)  document.getElementById(opts.pctId).textContent  = data.pct + '%';
    }

    if (opts.onStep) opts.onStep(data);

    if (data.done || data.error) {
      es.close();
      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }
      if (opts.onDone) opts.onDone(data);
    }
  };

  es.onerror = function () {
    es.close();
    if (opts.msgId) document.getElementById(opts.msgId).textContent = 'Connection lost.';
  };
}
