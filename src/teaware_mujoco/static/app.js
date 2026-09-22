const state = { status: null, episode: null, modality: "rgb", frame: 0 };
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setBusy(busy, message = "") {
  ["randomize", "collect"].forEach((id) => { $(id).disabled = busy; });
  $("message").classList.remove("error");
  $("message").textContent = message;
}

function showError(error) {
  $("message").classList.add("error");
  $("message").textContent = error.message;
  setBusy(false, error.message);
  $("message").classList.add("error");
}

function refreshImage() {
  const camera = $("camera").value;
  if (!camera) return;
  let source;
  if (state.episode) {
    const frameId = String(state.frame).padStart(6, "0");
    source = `/api/episodes/${state.episode.manifest.episode_id}/image?frame=${frameId}&camera=${encodeURIComponent(camera)}&modality=${state.modality}`;
    $("frame-label").textContent = frameId;
  } else {
    source = `/api/live/${encodeURIComponent(camera)}/${state.modality}`;
    $("frame-label").textContent = "LIVE";
  }
  const separator = source.includes("?") ? "&" : "?";
  $("viewport").src = `${source}${separator}cache=${Date.now()}`;
}

async function loadStatus() {
  state.status = await api("/api/status");
  $("status").textContent = `${state.status.scene_profile} / ${state.status.episode_count} episodes / ${state.status.renderer.width}x${state.status.renderer.height}`;
  $("camera").innerHTML = state.status.cameras.map((name) => `<option>${name}</option>`).join("");
  const robots = state.status.robots.map((robot) => `${robot.id}:${robot.handedness}-${robot.hand}[${robot.hand_dof}]`).join("  /  ");
  $("scene-strip").textContent = `ROBOTS  ${robots}   OBJECTS  ${state.status.objects.join("  /  ")}   DATASET  ${state.status.dataset_root}`;
  if (state.status.current_seed === null) {
    await api("/api/randomize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: 0 }) });
  }
  refreshImage();
}

async function loadEpisodes() {
  const episodes = await api("/api/episodes");
  $("episodes").innerHTML = episodes.length ? episodes.map((episode) => `
    <button class="episode ${state.episode?.manifest.episode_id === episode.episode_id ? "active" : ""}" data-id="${episode.episode_id}">
      <span>${episode.episode_id}</span><span class="${episode.valid ? "valid" : "invalid"}">${episode.valid ? "VALID" : "INVALID"}</span>
      <small>seed ${episode.seed}</small><small>${episode.frame_count} frames</small>
    </button>`).join("") : `<small>暂无采集记录</small>`;
  document.querySelectorAll(".episode").forEach((button) => {
    button.addEventListener("click", () => selectEpisode(button.dataset.id));
  });
}

async function selectEpisode(id) {
  const detail = await api(`/api/episodes/${id}`);
  state.episode = detail;
  state.frame = 0;
  $("frame").max = detail.manifest.frames.length - 1;
  $("frame").value = 0;
  $("metadata").textContent = JSON.stringify({
    episode_id: detail.manifest.episode_id,
    created_at: detail.manifest.created_at,
    seed: detail.manifest.seed,
    cameras: Object.keys(detail.manifest.cameras),
    scene_profile: detail.manifest.scene_profile,
    robots: detail.manifest.robots,
    objects: detail.manifest.objects,
    validation_errors: detail.validation_errors,
    config_sha256: detail.manifest.config_sha256,
  }, null, 2);
  await loadEpisodes();
  refreshImage();
}

$("modality").addEventListener("click", (event) => {
  if (!event.target.dataset.value) return;
  state.modality = event.target.dataset.value;
  document.querySelectorAll("#modality button").forEach((button) => button.classList.toggle("active", button === event.target));
  refreshImage();
});
$("camera").addEventListener("change", refreshImage);
$("frame").addEventListener("input", () => { state.frame = Number($("frame").value); refreshImage(); });
$("refresh").addEventListener("click", loadEpisodes);

$("randomize").addEventListener("click", async () => {
  try {
    setBusy(true, "正在重置场景");
    state.episode = null;
    $("frame").max = 0;
    await api("/api/randomize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: Number($("seed").value) }) });
    refreshImage();
    setBusy(false, "场景已更新");
  } catch (error) { showError(error); }
});

$("collect").addEventListener("click", async () => {
  try {
    setBusy(true, "采集中，请保持页面打开");
    const result = await api("/api/collect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: Number($("seed").value), episodes: Number($("count").value) }) });
    await loadStatus();
    await loadEpisodes();
    await selectEpisode(result.episodes[result.episodes.length - 1]);
    setBusy(false, `完成 ${result.episodes.length} 个 episode`);
  } catch (error) { showError(error); }
});

Promise.all([loadStatus(), loadEpisodes()]).catch(showError);
