const percent = value => `${(value * 100).toFixed(1)}%`;
const score = value => Number(value).toFixed(3);

async function loadDashboard() {
  const [response, awsResponse] = await Promise.all([
    fetch("data/dashboard.json"),
    fetch("data/aws-evidence.json"),
  ]);
  if (!response.ok) throw new Error(`Dashboard data unavailable: ${response.status}`);
  if (!awsResponse.ok) throw new Error(`AWS evidence unavailable: ${awsResponse.status}`);
  const [data, aws] = await Promise.all([response.json(), awsResponse.json()]);
  const selected = data.models.find(model => model.name === data.summary.selected_model);

  document.querySelector("#heroRate").textContent = percent(data.summary.default_rate);
  document.querySelector("#scoredRows").textContent = data.summary.scored_rows.toLocaleString("en-US");
  document.querySelector("#selectedModel").textContent = data.summary.selected_model.replaceAll("_", " ");
  document.querySelector("#selectedRoc").textContent = score(selected.roc_auc);
  document.querySelector("#selectedPr").textContent = score(selected.pr_auc);
  document.querySelector("#disclaimer").textContent = data.disclaimer;

  const comparison = document.querySelector("#modelComparison");
  data.models.forEach(model => {
    const row = document.createElement("div");
    row.className = `model-row ${model.name === data.summary.selected_model ? "selected" : ""}`;
    row.innerHTML = `<strong>${model.name.replaceAll("_", " ")}</strong><div class="track"><i style="width:${model.pr_auc * 100}%"></i></div><span>${score(model.pr_auc)}</span>`;
    comparison.appendChild(row);
  });

  const riskBands = document.querySelector("#riskBands");
  const detail = document.querySelector("#bandDetail");
  data.risk_bands.forEach((band, index) => {
    const button = document.createElement("button");
    button.className = `band ${index === 0 ? "active" : ""}`;
    button.innerHTML = `<span>${band.band}</span><strong>${band.clients.toLocaleString("en-US")}</strong><small>clients</small>`;
    button.addEventListener("click", () => {
      document.querySelectorAll(".band").forEach(item => item.classList.remove("active"));
      button.classList.add("active");
      detail.textContent = `${band.band.toUpperCase()} · average predicted probability ${percent(band.average_probability)} · observed default rate ${percent(band.observed_default_rate)}`;
    });
    riskBands.appendChild(button);
  });
  riskBands.querySelector("button").click();

  const calibration = document.querySelector("#calibration");
  data.calibration.forEach((bin, index) => {
    const group = document.createElement("div");
    group.className = "cal-group";
    group.innerHTML = `<i style="height:${Math.max(2, bin.average_probability * 100)}%" title="Predicted ${percent(bin.average_probability)}"></i><i class="observed" style="height:${Math.max(2, bin.observed_default_rate * 100)}%" title="Observed ${percent(bin.observed_default_rate)}"></i><small>${index + 1}</small>`;
    calibration.appendChild(group);
  });

  document.querySelector("#awsTraining").textContent = `${aws.training_seconds}s`;
  document.querySelector("#awsTransform").textContent = `${aws.transform_seconds}s`;
  document.querySelector("#awsPr").textContent = score(aws.pr_auc);
  document.querySelector("#awsRows").textContent = `${aws.test_rows.toLocaleString("en-US")} rows`;
  document.querySelector("#awsCleanup").textContent = aws.cleanup_complete ? "complete" : "pending";
}

loadDashboard().catch(error => {
  document.querySelector("#disclaimer").textContent = error.message;
});
