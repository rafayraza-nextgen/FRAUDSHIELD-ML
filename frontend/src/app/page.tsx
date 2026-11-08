"use client";

import React, { useState, useEffect } from "react";
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  Activity,
  UploadCloud,
  FileSpreadsheet,
  Moon,
  Sun,
  Zap,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  BarChart3,
  Layers,
  ArrowRight,
  RefreshCw,
  Sliders,
  DollarSign,
  Scale,
} from "lucide-react";
import {
  fetchHealth,
  fetchModelInfo,
  scoreSingleTransaction,
  uploadBatchCsv,
  TransactionPayload,
  PredictionResult,
  BatchPredictionResult,
  ModelInfo,
  SystemHealth,
} from "@/lib/api";

export default function Dashboard() {
  // Theme state
  const [isDark, setIsDark] = useState<boolean>(true);

  // Tab navigation
  const [activeTab, setActiveTab] = useState<"single" | "batch" | "benchmarks" | "xai">("single");

  // Health and Model info state
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isConnecting, setIsConnecting] = useState<boolean>(true);

  // Single transaction form state
  const [form, setForm] = useState<TransactionPayload>({
    step: 24,
    type: "TRANSFER",
    amount: 150000,
    oldbalanceOrg: 150000,
    newbalanceOrig: 0,
    oldbalanceDest: 0,
    newbalanceDest: 150000,
    nameOrig: "C198273641",
    nameDest: "C998127364",
  });
  const [scoring, setScoring] = useState<boolean>(false);
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  // Batch CSV state
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchUploading, setBatchUploading] = useState<boolean>(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResult | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // Initialize theme and health check
  useEffect(() => {
    const isDarkMode = document.documentElement.classList.contains("dark");
    setIsDark(isDarkMode);

    const initData = async () => {
      try {
        setIsConnecting(true);
        const [h, m] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchModelInfo().catch(() => null),
        ]);
        setHealth(h);
        setModelInfo(m);
      } catch (err) {
        console.error("Initialization error:", err);
      } finally {
        setIsConnecting(false);
      }
    };
    initData();
  }, []);

  const toggleTheme = () => {
    if (isDark) {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("fraudshield-theme", "light");
      setIsDark(false);
    } else {
      document.documentElement.classList.add("dark");
      localStorage.setItem("fraudshield-theme", "dark");
      setIsDark(true);
    }
  };

  // Form balance calculations
  const errorBalanceOrig = form.newbalanceOrig + form.amount - form.oldbalanceOrg;
  const errorBalanceDest = form.oldbalanceDest + form.amount - form.newbalanceDest;

  // Presets
  const applyPreset = (preset: "legit" | "fraud" | "merchant" | "cashout") => {
    switch (preset) {
      case "legit":
        setForm({
          step: 12,
          type: "TRANSFER",
          amount: 250.0,
          oldbalanceOrg: 1500.0,
          newbalanceOrig: 1250.0,
          oldbalanceDest: 5000.0,
          newbalanceDest: 5250.0,
          nameOrig: "C29182371",
          nameDest: "C83726194",
        });
        break;
      case "fraud":
        setForm({
          step: 48,
          type: "TRANSFER",
          amount: 850000.0,
          oldbalanceOrg: 850000.0,
          newbalanceOrig: 0.0,
          oldbalanceDest: 0.0,
          newbalanceDest: 0.0,
          nameOrig: "C19284712",
          nameDest: "C74829103",
        });
        break;
      case "merchant":
        setForm({
          step: 18,
          type: "PAYMENT",
          amount: 79.99,
          oldbalanceOrg: 2400.0,
          newbalanceOrig: 2320.01,
          oldbalanceDest: 0.0,
          newbalanceDest: 0.0,
          nameOrig: "C11223344",
          nameDest: "M99887766",
        });
        break;
      case "cashout":
        setForm({
          step: 72,
          type: "CASH_OUT",
          amount: 450000.0,
          oldbalanceOrg: 450000.0,
          newbalanceOrig: 0.0,
          oldbalanceDest: 25000.0,
          newbalanceDest: 475000.0,
          nameOrig: "C55443322",
          nameDest: "C99001122",
        });
        break;
    }
  };

  // Run single prediction
  const handleScore = async (e: React.FormEvent) => {
    e.preventDefault();
    setScoring(true);
    setSingleError(null);
    try {
      const res = await scoreSingleTransaction(form);
      setSingleResult(res);
    } catch (err: any) {
      setSingleError(err.message || "Failed to score transaction");
    } finally {
      setScoring(false);
    }
  };

  // Handle batch CSV
  const handleBatchUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFile) return;
    setBatchUploading(true);
    setBatchError(null);
    try {
      const res = await uploadBatchCsv(batchFile);
      setBatchResult(res);
    } catch (err: any) {
      setBatchError(err.message || "Batch scoring failed");
    } finally {
      setBatchUploading(false);
    }
  };

  // Download Sample CSV
  const downloadSampleCsv = () => {
    const csvHeader = "step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,nameOrig,nameDest\n";
    const sampleRows = [
      "1,TRANSFER,150000.0,150000.0,0.0,0.0,0.0,C12345,C98765\n",
      "2,PAYMENT,45.50,1200.0,1154.50,0.0,0.0,C23456,M34567\n",
      "3,CASH_OUT,250000.0,250000.0,0.0,10000.0,260000.0,C34567,C45678\n",
      "4,DEBIT,120.0,500.0,380.0,100.0,220.0,C45678,C56789\n",
      "5,CASH_IN,5000.0,1000.0,6000.0,10000.0,5000.0,C56789,C67890\n",
    ].join("");
    const blob = new Blob([csvHeader + sampleRows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "fraudshield_sample_transactions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen text-slate-800 dark:text-slate-100 flex flex-col selection:bg-brand-500 selection:text-white">
      {/* Top Cyber Navigation Bar */}
      <header className="sticky top-0 z-50 glass-panel border-b border-surface-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-cyber-cyan flex items-center justify-center text-white shadow-cyber">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <span className="font-extrabold tracking-wider text-lg sm:text-xl bg-clip-text text-transparent bg-gradient-to-r from-brand-600 via-cyber-cyan to-cyber-purple dark:from-blue-400 dark:via-cyan-300 dark:to-purple-400">
                F.R.A.U.D.S.H.I.E.L.D.
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-brand-100 text-brand-800 dark:bg-blue-950 dark:text-cyan-300 border border-brand-200 dark:border-cyan-800/50">
                v1.0 ML Engine
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* System Health Status */}
            <div className="flex items-center space-x-2 text-xs font-medium px-3 py-1.5 rounded-full bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700">
              <span
                className={`w-2 h-2 rounded-full ${
                  health?.model_loaded ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              <span className="text-slate-600 dark:text-slate-300">
                {isConnecting
                  ? "Connecting..."
                  : health?.model_loaded
                  ? `Active • RF (t*=${health.optimal_threshold})`
                  : "Connecting to API"}
              </span>
            </div>

            {/* Light / Dark Mode Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors border border-slate-200 dark:border-slate-700"
              title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-600" />}
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 sm:space-x-4 border-t border-slate-200/50 dark:border-slate-800/50 overflow-x-auto">
          {[
            { id: "single", label: "Real-Time Scoring", icon: Zap },
            { id: "batch", label: "Batch CSV Audit", icon: UploadCloud },
            { id: "xai", label: "SHAP Explainability", icon: HelpCircle },
            { id: "benchmarks", label: "Model Benchmarks", icon: BarChart3 },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-3 px-3 text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
                  active
                    ? "border-brand-600 text-brand-600 dark:border-cyber-cyan dark:text-cyber-cyan font-semibold"
                    : "border-transparent text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* TAB 1: REAL-TIME TRANSACTION SCORING */}
        {activeTab === "single" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Form Column */}
            <div className="lg:col-span-7 space-y-6">
              <div className="glass-panel p-6 rounded-2xl border border-surface-border shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-bold flex items-center space-x-2">
                      <Sliders className="w-5 h-5 text-brand-600 dark:text-cyber-cyan" />
                      <span>Transaction Parameters</span>
                    </h2>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Input transaction parameters or load pre-configured simulation scenarios.
                    </p>
                  </div>
                </div>

                {/* Preset Scenarios */}
                <div className="mb-6">
                  <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
                    Quick Simulation Presets:
                  </label>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <button
                      type="button"
                      onClick={() => applyPreset("legit")}
                      className="px-3 py-1.5 text-xs rounded-lg font-medium border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 transition"
                    >
                      Safe Transfer
                    </button>
                    <button
                      type="button"
                      onClick={() => applyPreset("fraud")}
                      className="px-3 py-1.5 text-xs rounded-lg font-medium border border-rose-300 dark:border-rose-800 bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300 hover:bg-rose-100 transition"
                    >
                      Account Drain Fraud
                    </button>
                    <button
                      type="button"
                      onClick={() => applyPreset("merchant")}
                      className="px-3 py-1.5 text-xs rounded-lg font-medium border border-blue-300 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 hover:bg-blue-100 transition"
                    >
                      Merchant Payment
                    </button>
                    <button
                      type="button"
                      onClick={() => applyPreset("cashout")}
                      className="px-3 py-1.5 text-xs rounded-lg font-medium border border-purple-300 dark:border-purple-800 bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300 hover:bg-purple-100 transition"
                    >
                      Suspicious Cash-Out
                    </button>
                  </div>
                </div>

                <form onSubmit={handleScore} className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Transaction Type */}
                    <div>
                      <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                        Transaction Type
                      </label>
                      <select
                        value={form.type}
                        onChange={(e) => setForm({ ...form, type: e.target.value as any })}
                        className="w-full px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-sm focus:ring-2 focus:ring-brand-500 outline-none"
                      >
                        <option value="TRANSFER">TRANSFER (High Risk Category)</option>
                        <option value="CASH_OUT">CASH_OUT (High Risk Category)</option>
                        <option value="PAYMENT">PAYMENT (Standard Consumer)</option>
                        <option value="DEBIT">DEBIT (Standard Consumer)</option>
                        <option value="CASH_IN">CASH_IN (Account Deposit)</option>
                      </select>
                    </div>

                    {/* Step (Simulation Hour) */}
                    <div>
                      <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                        Simulation Step (Hour 1 to 744)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="744"
                        value={form.step}
                        onChange={(e) => setForm({ ...form, step: parseInt(e.target.value) || 1 })}
                        className="w-full px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-sm focus:ring-2 focus:ring-brand-500 outline-none"
                      />
                    </div>
                  </div>

                  {/* Amount */}
                  <div>
                    <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                      Transaction Amount ($)
                    </label>
                    <div className="relative">
                      <DollarSign className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={form.amount}
                        onChange={(e) => setForm({ ...form, amount: parseFloat(e.target.value) || 0 })}
                        className="w-full pl-9 pr-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-sm focus:ring-2 focus:ring-brand-500 outline-none font-mono"
                      />
                    </div>
                  </div>

                  {/* Origin Account Balances */}
                  <div className="p-3 rounded-xl bg-slate-50/70 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
                      Sender Account (Origin)
                    </span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-slate-500 mb-1">Balance Before ($)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={form.oldbalanceOrg}
                          onChange={(e) => setForm({ ...form, oldbalanceOrg: parseFloat(e.target.value) || 0 })}
                          className="w-full px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-mono outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-slate-500 mb-1">Balance After ($)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={form.newbalanceOrig}
                          onChange={(e) => setForm({ ...form, newbalanceOrig: parseFloat(e.target.value) || 0 })}
                          className="w-full px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-mono outline-none"
                        />
                      </div>
                    </div>
                    {/* Live Derived Feature */}
                    <div className="mt-2 text-xs flex justify-between text-slate-500">
                      <span>Engineered Discrepancy (`errorBalanceOrig`):</span>
                      <span className={`font-mono font-bold ${Math.abs(errorBalanceOrig) > 1 ? "text-rose-500" : "text-emerald-500"}`}>
                        ${errorBalanceOrig.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {/* Destination Account Balances */}
                  <div className="p-3 rounded-xl bg-slate-50/70 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
                      Recipient Account (Destination)
                    </span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-slate-500 mb-1">Balance Before ($)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={form.oldbalanceDest}
                          onChange={(e) => setForm({ ...form, oldbalanceDest: parseFloat(e.target.value) || 0 })}
                          className="w-full px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-mono outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-slate-500 mb-1">Balance After ($)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={form.newbalanceDest}
                          onChange={(e) => setForm({ ...form, newbalanceDest: parseFloat(e.target.value) || 0 })}
                          className="w-full px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-sm font-mono outline-none"
                        />
                      </div>
                    </div>
                    <div className="mt-2 text-xs flex justify-between text-slate-500">
                      <span>Engineered Discrepancy (`errorBalanceDest`):</span>
                      <span className={`font-mono font-bold ${Math.abs(errorBalanceDest) > 1 ? "text-amber-500" : "text-emerald-500"}`}>
                        ${errorBalanceDest.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {singleError && (
                    <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-xs flex items-center space-x-2">
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                      <span>{singleError}</span>
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={scoring}
                    className="w-full py-3 px-4 rounded-xl font-bold text-white bg-gradient-to-r from-brand-600 to-cyber-cyan hover:from-brand-700 hover:to-cyan-600 shadow-cyber transition flex items-center justify-center space-x-2 disabled:opacity-50"
                  >
                    {scoring ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Evaluating ML Risk & SHAP Attributions...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" />
                        <span>Score Transaction with Champion Random Forest</span>
                      </>
                    )}
                  </button>
                </form>
              </div>
            </div>

            {/* Prediction Output & Risk Gauge Column */}
            <div className="lg:col-span-5 space-y-6">
              {singleResult ? (
                <div className="glass-panel p-6 rounded-2xl border border-surface-border shadow-sm space-y-6 animate-fade-in">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                      Evaluation Outcome
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      Latency: {singleResult.latency_ms}ms
                    </span>
                  </div>

                  {/* Circular Risk Score Display */}
                  <div className="text-center py-4">
                    <div
                      className={`inline-flex flex-col items-center justify-center w-36 h-36 rounded-full border-4 shadow-lg ${
                        singleResult.risk_score >= 75
                          ? "border-rose-500 shadow-cyber-danger bg-rose-500/10 text-rose-500"
                          : singleResult.risk_score >= 50
                          ? "border-amber-500 bg-amber-500/10 text-amber-500"
                          : "border-emerald-500 bg-emerald-500/10 text-emerald-500"
                      }`}
                    >
                      <span className="text-4xl font-extrabold font-mono tracking-tighter">
                        {singleResult.risk_score}
                      </span>
                      <span className="text-xs font-bold uppercase tracking-wider mt-1">
                        Risk Score
                      </span>
                    </div>

                    <div className="mt-4">
                      <span
                        className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-extrabold tracking-wide uppercase ${
                          singleResult.risk_tier === "HIGH"
                            ? "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 border border-rose-300 dark:border-rose-800"
                            : singleResult.risk_tier === "MODERATE"
                            ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border border-amber-300 dark:border-amber-800"
                            : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800"
                        }`}
                      >
                        {singleResult.risk_tier === "HIGH" ? (
                          <ShieldAlert className="w-4 h-4" />
                        ) : singleResult.risk_tier === "MODERATE" ? (
                          <AlertTriangle className="w-4 h-4" />
                        ) : (
                          <ShieldCheck className="w-4 h-4" />
                        )}
                        <span>{singleResult.risk_tier} RISK TIER</span>
                      </span>
                    </div>

                    <div className="mt-2 text-xs font-medium text-slate-500">
                      Recommendation:{" "}
                      <span className="font-bold text-slate-800 dark:text-slate-200">
                        {singleResult.recommendation.replace(/_/g, " ")}
                      </span>
                    </div>
                  </div>

                  {/* Summary Metric Chips */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                      <span className="text-slate-400 block mb-1">Fraud Probability</span>
                      <span className="text-base font-bold font-mono">
                        {(singleResult.fraud_probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                      <span className="text-slate-400 block mb-1">Calibrated Threshold</span>
                      <span className="text-base font-bold font-mono">
                        {(singleResult.threshold_applied * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* SHAP Summary Narrative */}
                  {singleResult.explanation?.human_readable_summary && (
                    <div className="p-4 rounded-xl bg-brand-50/50 dark:bg-slate-900/80 border border-brand-200 dark:border-slate-800 text-xs leading-relaxed space-y-2">
                      <div className="font-bold text-brand-800 dark:text-cyan-300 flex items-center space-x-1">
                        <Activity className="w-4 h-4" />
                        <span>AI Risk Attribution Summary:</span>
                      </div>
                      <p className="text-slate-600 dark:text-slate-300">
                        {singleResult.explanation.human_readable_summary}
                      </p>
                    </div>
                  )}

                  {/* Top SHAP Drivers */}
                  {singleResult.explanation?.risk_drivers?.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                        Top Factors Elevating Risk:
                      </span>
                      <div className="space-y-1.5">
                        {singleResult.explanation.risk_drivers.map((driver, idx) => (
                          <div
                            key={idx}
                            className="flex items-center justify-between p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs"
                          >
                            <span className="font-mono font-medium">{driver.feature}</span>
                            <span className="text-rose-600 dark:text-rose-400 font-bold font-mono">
                              +{driver.shap_value.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="glass-panel p-8 rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 text-center flex flex-col items-center justify-center space-y-3 min-h-[350px]">
                  <div className="w-12 h-12 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-400">
                    <Zap className="w-6 h-6" />
                  </div>
                  <h3 className="font-bold text-slate-700 dark:text-slate-200">Awaiting Transaction</h3>
                  <p className="text-xs text-slate-400 max-w-xs">
                    Configure the transaction parameters on the left and click "Score Transaction" to inspect real-time risk scores and SHAP feature attributions.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: BATCH CSV AUDIT */}
        {activeTab === "batch" && (
          <div className="space-y-8">
            <div className="glass-panel p-6 rounded-2xl border border-surface-border">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h2 className="text-lg font-bold flex items-center space-x-2">
                    <FileSpreadsheet className="w-5 h-5 text-cyber-purple" />
                    <span>High-Throughput Batch CSV Audit</span>
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Upload financial transaction logs for automated bulk risk evaluation and fraud categorization.
                  </p>
                </div>
                <button
                  onClick={downloadSampleCsv}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 transition"
                >
                  Download Sample CSV
                </button>
              </div>

              {/* Upload Zone */}
              <form onSubmit={handleBatchUpload} className="space-y-4">
                <div className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-2xl p-8 text-center hover:border-brand-500 dark:hover:border-cyber-cyan transition-colors cursor-pointer relative">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="flex flex-col items-center justify-center space-y-3">
                    <div className="w-12 h-12 rounded-xl bg-brand-50 dark:bg-slate-800 text-brand-600 dark:text-cyber-cyan flex items-center justify-center">
                      <UploadCloud className="w-6 h-6" />
                    </div>
                    <div>
                      <span className="font-semibold text-sm">
                        {batchFile ? batchFile.name : "Choose CSV file or drag and drop here"}
                      </span>
                      <p className="text-xs text-slate-400 mt-1">
                        Requires columns: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
                      </p>
                    </div>
                  </div>
                </div>

                {batchError && (
                  <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-xs">
                    {batchError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={!batchFile || batchUploading}
                  className="w-full py-3 px-4 rounded-xl font-bold text-white bg-gradient-to-r from-cyber-purple to-brand-600 hover:opacity-90 shadow-cyber-purple transition flex items-center justify-center space-x-2 disabled:opacity-40"
                >
                  {batchUploading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Auditing Batch Transactions...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Audit Batch File Now</span>
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Batch Results Overview */}
            {batchResult && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="glass-panel p-5 rounded-2xl border border-surface-border">
                    <span className="text-xs text-slate-400 block mb-1">Total Scored</span>
                    <span className="text-2xl font-extrabold font-mono">{batchResult.total_transactions}</span>
                  </div>
                  <div className="glass-panel p-5 rounded-2xl border border-surface-border">
                    <span className="text-xs text-rose-500 block mb-1">Flagged Fraud</span>
                    <span className="text-2xl font-extrabold font-mono text-rose-500">
                      {batchResult.fraud_detected_count}
                    </span>
                  </div>
                  <div className="glass-panel p-5 rounded-2xl border border-surface-border">
                    <span className="text-xs text-amber-500 block mb-1">Fraud Rate</span>
                    <span className="text-2xl font-extrabold font-mono text-amber-500">
                      {batchResult.fraud_rate_pct}%
                    </span>
                  </div>
                  <div className="glass-panel p-5 rounded-2xl border border-surface-border">
                    <span className="text-xs text-slate-400 block mb-1">Processing Time</span>
                    <span className="text-2xl font-extrabold font-mono text-cyan-500">
                      {batchResult.processing_time_sec}s
                    </span>
                  </div>
                </div>

                {/* Scored Transactions Table */}
                <div className="glass-panel rounded-2xl border border-surface-border overflow-hidden">
                  <div className="p-4 border-b border-surface-border flex justify-between items-center">
                    <h3 className="font-bold text-sm">Batch Scored Transactions</h3>
                    <span className="text-xs text-slate-400">
                      Showing top {Math.min(batchResult.predictions.length, 100)} records
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-100/50 dark:bg-slate-900/60 text-slate-500 uppercase tracking-wider font-semibold">
                        <tr>
                          <th className="p-3">Tx ID</th>
                          <th className="p-3">Probability</th>
                          <th className="p-3">Risk Score</th>
                          <th className="p-3">Risk Tier</th>
                          <th className="p-3">Fraud Decision</th>
                          <th className="p-3">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono">
                        {batchResult.predictions.slice(0, 50).map((tx, idx) => (
                          <tr
                            key={idx}
                            className={`hover:bg-slate-50/50 dark:hover:bg-slate-800/40 ${
                              tx.is_fraud ? "bg-rose-500/5" : ""
                            }`}
                          >
                            <td className="p-3 font-semibold">{tx.transaction_id}</td>
                            <td className="p-3">{(tx.fraud_probability * 100).toFixed(2)}%</td>
                            <td className="p-3 font-bold">{tx.risk_score}</td>
                            <td className="p-3">
                              <span
                                className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                  tx.risk_tier === "HIGH"
                                    ? "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300"
                                    : tx.risk_tier === "MODERATE"
                                    ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                                    : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                                }`}
                              >
                                {tx.risk_tier}
                              </span>
                            </td>
                            <td className="p-3">
                              {tx.is_fraud ? (
                                <span className="text-rose-500 font-bold">FRAUD</span>
                              ) : (
                                <span className="text-emerald-500 font-semibold">LEGIT</span>
                              )}
                            </td>
                            <td className="p-3 text-slate-400 font-sans">{tx.recommendation}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: SHAP EXPLAINABILITY */}
        {activeTab === "xai" && (
          <div className="space-y-6">
            <div className="glass-panel p-6 rounded-2xl border border-surface-border">
              <h2 className="text-lg font-bold flex items-center space-x-2">
                <HelpCircle className="w-5 h-5 text-brand-600 dark:text-cyber-cyan" />
                <span>Explainable AI (SHAP) Principles & Methodology</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
                Financial fraud detection models cannot be black boxes. F.R.A.U.D.S.H.I.E.L.D. implements game-theoretic Shapley values (via TreeSHAP) to attribute precise marginal contributions to each engineered feature.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <div className="font-bold text-sm mb-1 text-brand-600 dark:text-cyan-400">1. Balance Discrepancies</div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Engineered features `errorBalanceOrig` and `errorBalanceDest` contribute over 40% of predictive power, pinpointing abnormal zeroing of source funds.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <div className="font-bold text-sm mb-1 text-purple-600 dark:text-purple-400">2. Transaction Classification</div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Simulations prove that 100% of PaySim fraudulent transactions concentrate in `TRANSFER` and `CASH_OUT` operations, heavily weighted by decision trees.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <div className="font-bold text-sm mb-1 text-emerald-600 dark:text-emerald-400">3. Non-Discriminatory Features</div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Account identifiers (`nameOrig`, `nameDest`) and synthetic simulation flags (`isFlaggedFraud`) are explicitly removed during preprocessing to prevent data leakage.
                  </p>
                </div>
              </div>

              {/* Explainer Visual Card */}
              <div className="mt-8 p-6 rounded-2xl bg-gradient-to-r from-brand-900 to-slate-900 text-white border border-slate-800">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div>
                    <h3 className="font-bold text-base text-cyan-300">SHAP Waterfall & Global Attribution Artifacts</h3>
                    <p className="text-xs text-slate-300 mt-1">
                      Visual diagnostic plots (beeswarm, bar importance, waterfall charts) are persisted to `reports/figures/`.
                    </p>
                  </div>
                  <div className="flex space-x-2 text-xs font-mono">
                    <span className="px-3 py-1.5 rounded-lg bg-white/10 border border-white/20">12_shap_global_summary.png</span>
                    <span className="px-3 py-1.5 rounded-lg bg-white/10 border border-white/20">14_shap_waterfall.png</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: MODEL BENCHMARKS */}
        {activeTab === "benchmarks" && (
          <div className="space-y-6">
            <div className="glass-panel p-6 rounded-2xl border border-surface-border">
              <h2 className="text-lg font-bold flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-brand-600 dark:text-cyber-cyan" />
                <span>Multi-Model Comparative Evaluation & Metric Philosophy</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
                Under severe class imbalance (0.13% fraud), accuracy is a deceptive metric. A naive classifier predicting 100% legitimate transactions achieves 99.87% accuracy while missing every single fraud event. F.R.A.U.D.S.H.I.E.L.D. optimizes for <strong>PR-AUC</strong> and <strong>Recall at Tuned Threshold</strong>.
              </p>

              {/* KPI Highlights */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Champion Model</span>
                  <span className="text-lg font-bold">Random Forest</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Test PR-AUC</span>
                  <span className="text-lg font-bold font-mono text-cyan-500">0.9942</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Recall @ t*=0.09</span>
                  <span className="text-lg font-bold font-mono text-emerald-500">85.42%</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Inference Latency</span>
                  <span className="text-lg font-bold font-mono text-purple-500">&lt; 15 ms</span>
                </div>
              </div>

              {/* Benchmark Table */}
              <div className="mt-8 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100/60 dark:bg-slate-900/80 text-slate-500 uppercase tracking-wider font-semibold">
                    <tr>
                      <th className="p-3">Model Architecture</th>
                      <th className="p-3">Precision</th>
                      <th className="p-3">Recall</th>
                      <th className="p-3">F1 Score</th>
                      <th className="p-3">ROC-AUC</th>
                      <th className="p-3">PR-AUC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono">
                    <tr className="bg-brand-50/30 dark:bg-brand-950/20 font-semibold">
                      <td className="p-3 text-brand-600 dark:text-cyan-400">Random Forest (Champion)</td>
                      <td className="p-3">0.9524</td>
                      <td className="p-3">0.8542</td>
                      <td className="p-3">0.9006</td>
                      <td className="p-3">0.9996</td>
                      <td className="p-3 text-brand-600 dark:text-cyan-400">0.9942</td>
                    </tr>
                    <tr>
                      <td className="p-3">Decision Tree</td>
                      <td className="p-3">0.8912</td>
                      <td className="p-3">0.8250</td>
                      <td className="p-3">0.8568</td>
                      <td className="p-3">0.9120</td>
                      <td className="p-3">0.8410</td>
                    </tr>
                    <tr>
                      <td className="p-3">Logistic Regression</td>
                      <td className="p-3">0.6840</td>
                      <td className="p-3">0.5120</td>
                      <td className="p-3">0.5857</td>
                      <td className="p-3">0.8654</td>
                      <td className="p-3">0.5892</td>
                    </tr>
                    <tr>
                      <td className="p-3">K-Nearest Neighbors (KNN)</td>
                      <td className="p-3">0.7410</td>
                      <td className="p-3">0.4820</td>
                      <td className="p-3">0.5841</td>
                      <td className="p-3">0.8210</td>
                      <td className="p-3">0.5430</td>
                    </tr>
                    <tr>
                      <td className="p-3">Calibrated LinearSVC</td>
                      <td className="p-3">0.6920</td>
                      <td className="p-3">0.4950</td>
                      <td className="p-3">0.5772</td>
                      <td className="p-3">0.8490</td>
                      <td className="p-3">0.5612</td>
                    </tr>
                    <tr>
                      <td className="p-3">Gaussian Naive Bayes</td>
                      <td className="p-3">0.1240</td>
                      <td className="p-3">0.7810</td>
                      <td className="p-3">0.2140</td>
                      <td className="p-3">0.8340</td>
                      <td className="p-3">0.2450</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Cyber Fintech Footer */}
      <footer className="glass-panel border-t border-surface-border mt-auto py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-4">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-brand-600 dark:text-cyan-400" />
            <span className="font-semibold text-slate-600 dark:text-slate-300">
              F.R.A.U.D.S.H.I.E.L.D.
            </span>
            <span>— Financial Risk Assessment & Unauthorized Data Shield</span>
          </div>
          <div>
            <span>Production Grade ML Pipeline • 100% Free Tier Compliant</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
