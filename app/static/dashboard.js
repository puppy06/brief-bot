(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file");
  const btnRun = document.getElementById("btnRun");
  const modelInput = document.getElementById("model");
  const jobCard = document.getElementById("jobCard");
  const progressBar = document.getElementById("progressBar");
  const stepText = document.getElementById("stepText");
  const logList = document.getElementById("logList");
  const doneActions = document.getElementById("doneActions");
  const errText = document.getElementById("errText");
  const dlVideo = document.getElementById("dlVideo");
  const dlBrief = document.getElementById("dlBrief");

  let selectedFile = null;
  let pollTimer = null;

  function setFile(file) {
    const isPdf =
      !!file &&
      (file.type === "application/pdf" ||
        (typeof file.name === "string" &&
          file.name.toLowerCase().endsWith(".pdf")));
    if (!isPdf) {
      selectedFile = null;
      btnRun.disabled = true;
      return;
    }
    selectedFile = file;
    btnRun.disabled = false;
    dropzone.querySelector(".dropzone-inner p strong").textContent = file.name;
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    setFile(f);
    fileInput.files = e.dataTransfer.files;
  });

  function renderLogs(logs) {
    logList.innerHTML = "";
    (logs || []).forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      logList.appendChild(li);
    });
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function pollJob(jobId) {
    stopPoll();
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch("/api/jobs/" + encodeURIComponent(jobId));
        if (!r.ok) throw new Error("Status poll failed");
        const j = await r.json();
        stepText.textContent = j.step || j.status;
        const pct =
          typeof j.progress === "number" ? Math.round(j.progress * 100) : 5;
        progressBar.style.width = Math.min(100, pct) + "%";
        renderLogs(j.logs);
        if (j.status === "done") {
          stopPoll();
          progressBar.style.width = "100%";
          doneActions.classList.remove("hidden");
          dlVideo.href = "/api/jobs/" + encodeURIComponent(jobId) + "/video";
          dlBrief.href = "/api/jobs/" + encodeURIComponent(jobId) + "/brief";
          errText.hidden = true;
        }
        if (j.status === "error") {
          stopPoll();
          errText.hidden = false;
          errText.textContent = j.error || "Job failed.";
        }
      } catch (e) {
        stopPoll();
        errText.hidden = false;
        errText.textContent = String(e);
      }
    }, 1200);
  }

  btnRun.addEventListener("click", async () => {
    if (!selectedFile) return;
    errText.hidden = true;
    doneActions.classList.add("hidden");
    jobCard.classList.remove("hidden");
    progressBar.style.width = "5%";
    stepText.textContent = "Starting…";
    logList.innerHTML = "";

    const fd = new FormData();
    fd.append("file", selectedFile);
    const model = modelInput.value.trim() || "llama3.2";
    const q = "?model=" + encodeURIComponent(model);

    try {
      const r = await fetch("/api/jobs" + q, { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      const j = await r.json();
      pollJob(j.job_id);
    } catch (e) {
      errText.hidden = false;
      errText.textContent = String(e);
    }
  });
})();
