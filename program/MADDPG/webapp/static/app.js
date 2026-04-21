async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `请求失败: ${response.status}`);
  }
  return data;
}

function parseJSONText(text, fallback) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function show(elId, content) {
  const el = document.getElementById(elId);
  el.textContent = typeof content === "string" ? content : JSON.stringify(content, null, 2);
}

document.getElementById("btnLoadNav").onclick = async () => {
  const filePath = document.getElementById("navPath").value.trim();
  try {
    const res = await postJSON("/api/navigation/load", { file_path: filePath });
    show("navResult", res.data);
  } catch (error) {
    show("navResult", String(error));
  }
};

document.getElementById("btnPlan").onclick = async () => {
  const start = parseJSONText(document.getElementById("start").value, [1, 2]);
  const goal = parseJSONText(document.getElementById("goal").value, [28, 12]);
  const via = parseJSONText(document.getElementById("via").value, []);
  try {
    const res = await postJSON("/api/navigation/plan", { start, goal, via });
    show("planResult", res.data);
  } catch (error) {
    show("planResult", String(error));
  }
};

document.getElementById("btnLoadLLM").onclick = async () => {
  const model_id = document.getElementById("modelId").value.trim();
  const model_path = document.getElementById("modelPath").value.trim();
  try {
    const res = await postJSON("/api/llm/load", {
      model_id,
      model_path: model_path || null,
    });
    show("llmResult", res.data);
  } catch (error) {
    show("llmResult", String(error));
  }
};

document.getElementById("btnTune").onclick = async () => {
  const dataset_path = document.getElementById("datasetPath").value.trim();
  const config = parseJSONText(document.getElementById("tuneConfig").value, {});
  try {
    const res = await postJSON("/api/llm/fine_tune", { dataset_path, config });
    show("llmResult", res.data);
  } catch (error) {
    show("llmResult", String(error));
  }
};

document.getElementById("btnSimMetrics").onclick = async () => {
  const epochs = Number(document.getElementById("simEpochs").value) || 30;
  const seed = Number(document.getElementById("simSeed").value) || 42;
  const loss_noise = Number(document.getElementById("simLossNoise").value) || 0.04;
  const metric_noise = Number(document.getElementById("simMetricNoise").value) || 0.012;

  try {
    const res = await postJSON("/api/llm/simulate_metrics", {
      config: { epochs, seed, loss_noise, metric_noise },
    });
    show("simResult", res.data);

    if (res.data.image_url) {
      const img = document.getElementById("simMetricsImage");
      img.src = `${res.data.image_url}?t=${Date.now()}`;
      img.style.display = "block";
    }
  } catch (error) {
    show("simResult", String(error));
  }
};

document.getElementById("btnDashboard").onclick = async () => {
  const groups = parseJSONText(document.getElementById("groups").value, []);
  const title = document.getElementById("title").value.trim();
  try {
    const res = await postJSON("/api/dashboard/run", { groups, title });
    show("runResult", res.data);
    if (res.data.image_url) {
      const img = document.getElementById("dashboardImage");
      img.src = `${res.data.image_url}?t=${Date.now()}`;
      img.style.display = "block";
    }
  } catch (error) {
    show("runResult", String(error));
  }
};

document.getElementById("btnExisting").onclick = async () => {
  try {
    const res = await postJSON("/api/features/run_existing", {});
    show("runResult", res.data);
  } catch (error) {
    show("runResult", String(error));
  }
};
