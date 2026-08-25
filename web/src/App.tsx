import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import { 
  Scan, 
  UserPlus, 
  Users, 
  Settings, 
  CheckCircle2, 
  XCircle, 
  Camera, 
  Trash2, 
  Search, 
  Plus, 
  Sparkles, 
  Database, 
  ArrowRight, 
  RefreshCw, 
  Clock, 
  CreditCard, 
  Zap, 
  Star, 
  Compass, 
  Lock, 
  Smartphone, 
  Coins,
  AlertTriangle,
  Timer
} from 'lucide-react';

interface User {
  id?: number;
  username: string;
  sample_count: number;
  enrolled_at: string;
}

interface ScanResult {
  accepted: boolean;
  username: string | null;
  score: number;
  threshold: number;
  time_ms: number;
  clahe_base64?: string;
  action_type?: string;
}

interface ReportData {
  self_matches?: Array<[string, number, number, number, string]>;
  cross_matches?: Array<[string, number, string]>;
}

// ── Custom Palm Vein SVG Icon (Clean 5-finger palm with sub-dermal vein tracks) ──
function PalmIcon({ className = "w-12 h-12 text-black", animated = false }: { className?: string; animated?: boolean }) {
  return (
    <svg 
      viewBox="0 0 100 100" 
      fill="currentColor" 
      className={`${className} ${animated ? 'animate-pulse' : ''}`}
    >
      {/* Palm Base & 5 Fingers Outline */}
      <path 
        d="M28 42 C28 32, 33 32, 33 42 L33 55 C33 57, 36 57, 36 55 L36 28 C36 18, 42 18, 42 28 L42 53 C42 55, 45 55, 45 53 L45 22 C45 12, 51 12, 51 22 L51 53 C51 55, 54 55, 54 53 L54 30 C54 20, 60 20, 60 30 L60 58 C60 60, 63 60, 63 58 L65 46 C67 38, 74 40, 72 49 L69 64 C65 78, 56 86, 46 86 C34 86, 26 76, 26 62 L26 42 Z" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="5" 
        strokeLinecap="round" 
        strokeLinejoin="round" 
      />
      {/* Sub-dermal Vein Pattern Nodes */}
      <path 
        d="M48 80 L48 65 M48 65 L38 52 M48 65 L58 52 M38 52 L38 40 M58 52 L58 40 M48 52 L48 35" 
        fill="none" 
        stroke={animated ? "#38BDF8" : "currentColor"} 
        strokeWidth="3.5" 
        strokeLinecap="round" 
        strokeDasharray={animated ? "2 3" : "none"}
      />
      {/* Biometric Sensor Points */}
      <circle cx="48" cy="65" r="3.5" fill="#FFDE59" stroke="#121212" strokeWidth="2" />
      <circle cx="38" cy="52" r="3" fill="#CCFF00" stroke="#121212" strokeWidth="1.5" />
      <circle cx="58" cy="52" r="3" fill="#CCFF00" stroke="#121212" strokeWidth="1.5" />
    </svg>
  );
}

const SAMPLE_GUIDANCE = [
  "Step 1: Hold palm flat, centered ~10-15cm above sensor",
  "Step 2: Tilt palm slightly to the LEFT (~5 degrees)",
  "Step 3: Tilt palm slightly to the RIGHT (~5 degrees)",
  "Step 4: Raise palm slightly HIGHER (~15-18cm)",
  "Step 5: Spread fingers slightly wider",
  "Step 6: Hold palm flat for final confirmation",
];

export default function App() {
  // Navigation: 'landing' | 'scan' | 'enroll' | 'users' | 'admin'
  const [activeTab, setActiveTab] = useState<'landing' | 'scan' | 'enroll' | 'users' | 'admin'>('landing');
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraType, setCameraType] = useState('Checking...');
  const [users, setUsers] = useState<User[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Authorization mode in Scan tab
  const [selectedAuthAction, setSelectedAuthAction] = useState<{ id: string; name: string; desc: string; icon: string }>({
    id: 'pay',
    name: 'Palm Pay Auth',
    desc: 'Payment Token',
    icon: '💳'
  });

  // Scanning State & Countdown
  const [isScanning, setIsScanning] = useState(false);
  const [scanCountdown, setScanCountdown] = useState<number | null>(null);
  const [lastScan, setLastScan] = useState<ScanResult | null>(null);
  const [resultOverlay, setResultOverlay] = useState<ScanResult | null>(null);

  // Enrollment State & 5-Second Countdown
  const [enrollUsername, setEnrollUsername] = useState('');
  const [enrollSamples, setEnrollSamples] = useState<Array<{ vr_mean: number; thumb: string }>>([]);
  const [isCapturingSample, setIsCapturingSample] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState<number | null>(null);
  const [enrollStatusMsg, setEnrollStatusMsg] = useState('');

  // Modals & Toasts
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'warn' | 'error' } | null>(null);

  const showToast = (msg: string, type: 'success' | 'warn' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  // Load Data from Backend
  const loadUsers = async () => {
    try {
      const res = await fetch('/api/users');
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users || []);
      }
    } catch (e) {
      console.warn('Backend offline:', e);
    }
  };

  const loadStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setCameraReady(data.camera_available ?? false);
        setCameraType(data.camera_type || 'None');
      }
    } catch (e) {
      setCameraReady(false);
      setCameraType('Disconnected');
    }
  };

  useEffect(() => {
    loadUsers();
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Cancel enrollment when leaving enroll tab with partial samples
  useEffect(() => {
    if (activeTab !== 'enroll' && enrollSamples.length > 0 && enrollUsername) {
      fetch('/api/enroll/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: enrollUsername }),
      }).catch(() => {});
      setEnrollSamples([]);
      setEnrollUsername('');
      setEnrollStatusMsg('');
    }
  }, [activeTab]);

  // Trigger Real Scan with 3-Second Countdown
  const handleScanWithCountdown = async (actionType = 'Palm Pay Auth') => {
    if (isScanning || scanCountdown !== null) return;
    
    // 3-Second countdown
    setScanCountdown(3);
    for (let i = 3; i > 0; i--) {
      setScanCountdown(i);
      await new Promise(r => setTimeout(r, 1000));
    }
    setScanCountdown(null);
    setIsScanning(true);

    try {
      const res = await fetch('/api/scan', { method: 'POST' });
      if (res.ok) {
        const data: ScanResult = await res.json();
        const finalRes: ScanResult = {
          ...data,
          action_type: actionType,
        };
        setLastScan(finalRes);
        setResultOverlay(finalRes);

        if (finalRes.accepted) {
          confetti({
            particleCount: 80,
            spread: 70,
            origin: { y: 0.6 },
            colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00', '#121212']
          });
        }
      } else {
        const err = await res.json();
        showToast(err.detail || 'Scan failed: Palm not detected', 'warn');
      }
    } catch {
      showToast('Cannot connect to server. Ensure server.py is running on Pi.', 'error');
    } finally {
      setIsScanning(false);
    }
  };

  // Live Camera Sample Capture with 5-Second Countdown
  const handleCaptureSampleWithCountdown = async () => {
    if (isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6) return;
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (!cleanUname) {
      showToast('Enter a username or ID first!', 'warn');
      return;
    }

    const currentHint = SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm steady ~10-15cm above sensor';
    setEnrollStatusMsg(`${currentHint} (Capturing in 5 seconds...)`);

    // 5-Second Timer Countdown
    for (let i = 5; i > 0; i--) {
      setEnrollCountdown(i);
      await new Promise(r => setTimeout(r, 1000));
    }
    setEnrollCountdown(null);
    setIsCapturingSample(true);
    setEnrollStatusMsg('📸 Snapping frame & computing Gabor VeinCode...');

    try {
      const res = await fetch('/api/enroll/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUname, sample_idx: enrollSamples.length }),
      });
      if (res.ok) {
        const data = await res.json();
        setEnrollSamples(prev => [...prev, { vr_mean: data.vr_mean || 0.5, thumb: data.thumb || '' }]);
        const nextHint = SAMPLE_GUIDANCE[enrollSamples.length + 1] || 'Ready for next capture';
        setEnrollStatusMsg(`Sample #${enrollSamples.length + 1} captured! Next: ${nextHint}`);
        showToast(`Sample ${enrollSamples.length + 1}/6 captured from camera!`, 'success');
      } else {
        const err = await res.json();
        const msg = err.detail || 'Hand not detected. Hold palm flat ~10-15cm above sensor.';
        setEnrollStatusMsg(`Capture failed: ${msg}`);
        showToast(msg, 'warn');
      }
    } catch {
      showToast('Server connection failed.', 'error');
      setEnrollStatusMsg('Error communicating with backend server.');
    } finally {
      setIsCapturingSample(false);
    }
  };

  // Save Enrollment to Database
  const handleSaveEnrollment = async () => {
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (enrollSamples.length < 3 || !cleanUname) {
      showToast('Capture at least 3 samples before saving!', 'warn');
      return;
    }
    try {
      const res = await fetch('/api/enroll/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUname }),
      });
      if (res.ok) {
        showToast(`ENROLLED '${cleanUname}' with ${enrollSamples.length} templates!`, 'success');
        confetti({
          particleCount: 100,
          spread: 80,
          origin: { y: 0.5 },
          colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00']
        });
        setEnrollUsername('');
        setEnrollSamples([]);
        setEnrollStatusMsg('');
        loadUsers();
        setActiveTab('users');
      } else {
        const err = await res.json();
        showToast(err.detail || 'Failed to save enrollment to database.', 'error');
      }
    } catch {
      showToast('Network error saving to database.', 'error');
    }
  };

  // Delete User
  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      const res = await fetch(`/api/users/${deleteTarget}`, { method: 'DELETE' });
      if (res.ok) {
        setUsers(prev => prev.filter(u => u.username !== deleteTarget));
        showToast(`User '${deleteTarget}' removed from database!`, 'success');
      } else {
        showToast('Failed to delete user.', 'error');
      }
    } catch {
      showToast('Error connecting to backend.', 'error');
    }
    setDeleteTarget(null);
  };

  // Open Report
  const openReport = async () => {
    setReportModalOpen(true);
    try {
      const res = await fetch('/api/report');
      if (res.ok) {
        const data = await res.json();
        setReportData(data);
      }
    } catch {
      setReportData(null);
    }
  };

  const filteredUsers = users.filter(u => 
    u.username.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const tabList: Array<{ id: 'landing' | 'scan' | 'enroll' | 'users' | 'admin'; label: string; icon: any }> = [
    { id: 'landing', label: 'Home', icon: Compass },
    { id: 'scan', label: 'Scan', icon: Scan },
    { id: 'enroll', label: 'Enroll', icon: UserPlus },
    { id: 'users', label: 'Friends', icon: Users },
    { id: 'admin', label: 'Stats', icon: Settings },
  ];

  const activeTabIdx = tabList.findIndex(t => t.id === activeTab);

  return (
    <div className="min-h-screen bg-dribbble-yellow flex justify-center items-center p-0 sm:p-6 text-[#121212] select-none font-sans">
      
      {/* ── PHONE CONTAINER ── */}
      <div className="w-full max-w-[480px] min-h-screen sm:min-h-[854px] sm:h-[854px] bg-[#FFFDF0] border-x-0 sm:border-[4px] border-black sm:rounded-[36px] sm:shadow-[10px_10px_0px_#121212] flex flex-col relative overflow-hidden bg-neo-cream">

        {/* ── TOP STATUS / SERVICE BAR ── */}
        <div className="px-6 pt-3 pb-1 flex items-center justify-between text-xs font-black text-black z-20">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-black"></span>
            <span className="w-2 h-2 rounded-full bg-black"></span>
            <span className="text-[11px] uppercase">{cameraType}</span>
          </div>
          <span className="font-display text-sm font-black">19:02</span>
          <div className="flex items-center gap-1.5">
            <div 
              title={cameraReady ? `Camera Active: ${cameraType}` : 'No Camera Detected'} 
              className={`w-3 h-3 rounded-full border-[1.5px] border-black ${cameraReady ? 'bg-[#CCFF00]' : 'bg-[#FF4081]'}`} 
            />
            <div className="w-5 h-2.5 border-[1.5px] border-black rounded-sm p-[1px]">
              <div className="w-full h-full bg-black rounded-[0.5px]"></div>
            </div>
          </div>
        </div>

        {/* ── TOP PROFILE / HEADER ── */}
        <header className="px-5 py-2.5 flex items-center justify-between z-20 border-b-[3px] border-black bg-[#FFFDF0]">
          <div 
            onClick={() => setActiveTab('landing')} 
            className="flex items-center gap-2.5 cursor-pointer neo-btn"
          >
            <div className="w-10 h-10 rounded-full bg-[#FFDE59] border-[2.5px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black text-base">
              <PalmIcon className="w-6 h-6 text-black" />
            </div>
            <div>
              <h1 className="font-display font-black text-base leading-none">Sam Smith</h1>
              <p className="text-[10px] font-bold text-[#666] mt-0.5">Palm Vein Biometrics</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="px-2.5 py-1 bg-[#38BDF8] border-[2px] border-black rounded-full shadow-[2px_2px_0px_#121212] text-xs font-black flex items-center gap-1">
              <Star className="w-3.5 h-3.5 fill-[#FFDE59] text-black" />
              <span>{users.length} Users</span>
            </div>

            <button 
              onClick={() => setActiveTab('landing')}
              className="w-8 h-8 rounded-full bg-[#FFDE59] border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black text-xs neo-btn"
              title="Home Landing"
            >
              ✏️
            </button>
          </div>
        </header>

        {/* ── TOAST ALERT ── */}
        {toast && (
          <div className="absolute top-20 left-6 right-6 z-50 animate-bounce">
            <div className={`p-3 border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-xs text-center flex items-center justify-center gap-2 ${
              toast.type === 'error' ? 'bg-[#FF4081] text-white' : toast.type === 'warn' ? 'bg-[#FF7A00] text-white' : 'bg-[#CCFF00] text-black'
            }`}>
              {toast.type === 'error' || toast.type === 'warn' ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
              <span>{toast.msg}</span>
            </div>
          </div>
        )}

        {/* ── MAIN SCROLLABLE CONTENT ── */}
        <main className="flex-1 overflow-y-auto px-5 py-4 pb-28 space-y-4">

          {/* ══════════════════════════════════════════════════════════
              PAGE 1: ANIMATED LANDING PAGE
             ══════════════════════════════════════════════════════════ */}
          {activeTab === 'landing' && (
            <div className="space-y-4 animate-fadeIn">
              
              {!cameraReady && (
                <div className="bg-[#FF7A00] text-white border-[3px] border-black rounded-2xl p-3 shadow-[3px_3px_0px_#121212] flex items-center gap-2.5 text-xs font-black">
                  <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                  <span>Camera not detected. Connect Raspberry Pi NoIR Camera.</span>
                </div>
              )}

              {/* Hero Card */}
              <div className="bg-white border-[3px] border-black rounded-3xl p-5 shadow-[6px_6px_0px_#121212] relative overflow-hidden flex flex-col items-center text-center">
                
                <div className="absolute top-2 left-3 w-8 h-8 border-[2px] border-black bg-[#FFDE59] rounded-md grid grid-cols-2 grid-rows-2">
                  <div className="border-r border-b border-black"></div>
                  <div className="border-b border-black"></div>
                  <div className="border-r border-black"></div>
                </div>

                <div className="absolute top-3 right-3 px-2 py-0.5 bg-[#38BDF8] border-[2px] border-black rounded-lg text-[10px] font-black shadow-[2px_2px_0px_#121212] animate-float">
                  ⏱️ 5s AUTO-TIMER
                </div>

                {/* Smartphone Terminal Card */}
                <div className="relative my-3 w-full flex items-center justify-center">
                  <div className="absolute -left-1 top-2 w-9 h-9 rounded-full bg-[#FFDE59] border-[2.5px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black text-sm animate-bounce">
                    🪙
                  </div>

                  <div className="relative w-28 h-36 bg-[#FFFDF0] border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] flex flex-col items-center justify-between p-2 overflow-hidden">
                    <div className="w-8 h-1 bg-black rounded-full mb-1"></div>
                    <div className="w-16 h-16 rounded-full bg-[#CCFF00] border-[2.5px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center animate-pulse">
                      <PalmIcon className="w-10 h-10 text-black" animated={true} />
                    </div>
                    <div className="w-full py-1 bg-[#38BDF8] border-[1.5px] border-black rounded-md text-[9px] font-black text-center uppercase tracking-tighter">
                      PALM VEIN POS
                    </div>
                  </div>

                  <div className="absolute -right-1 bottom-4 px-2.5 py-1 rounded-xl bg-[#FF4081] text-white border-[2px] border-black shadow-[2px_2px_0px_#121212] text-[10px] font-black flex items-center gap-1 animate-wiggle">
                    <Timer className="w-3 h-3 text-white" />
                    <span>5s Timer</span>
                  </div>
                </div>

                <div className="space-y-1 my-2">
                  <h2 className="font-display font-black text-xl leading-tight">
                    Sub-dermal infrared <span className="text-[#FF4081]">palm authentication</span>
                  </h2>
                  <p className="text-xs font-bold text-[#666]">5-second countdown timer for perfect multi-angle hand positioning.</p>
                </div>

                <button
                  onClick={() => setActiveTab('scan')}
                  className="mt-2 w-full py-3.5 bg-[#FFDE59] border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-base flex items-center justify-center gap-2 neo-btn hover:bg-[#ffe373]"
                >
                  <span>Open Palm Scanner</span>
                  <ArrowRight className="w-5 h-5 stroke-[3]" />
                </button>
              </div>

              {/* Quick Navigation Cards */}
              <div className="grid grid-cols-2 gap-3">
                <div 
                  onClick={() => setActiveTab('enroll')}
                  className="bg-[#38BDF8] border-[3px] border-black rounded-2xl p-3.5 shadow-[4px_4px_0px_#121212] cursor-pointer neo-btn"
                >
                  <div className="w-8 h-8 rounded-full bg-white border-[2px] border-black flex items-center justify-center font-black text-sm mb-2">
                    ➕
                  </div>
                  <h4 className="font-display font-black text-sm">Enroll Palm</h4>
                  <p className="text-[10px] font-bold text-black mt-0.5">5s timer per sample</p>
                </div>

                <div 
                  onClick={() => setActiveTab('users')}
                  className="bg-[#CCFF00] border-[3px] border-black rounded-2xl p-3.5 shadow-[4px_4px_0px_#121212] cursor-pointer neo-btn"
                >
                  <div className="w-8 h-8 rounded-full bg-white border-[2px] border-black flex items-center justify-center font-black text-sm mb-2">
                    👥
                  </div>
                  <h4 className="font-display font-black text-sm">Enrolled Users</h4>
                  <p className="text-[10px] font-bold text-black mt-0.5">{users.length} registered</p>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════
              PAGE 2: SCAN & PALM AUTHENTICATION TERMINAL
             ══════════════════════════════════════════════════════════ */}
          {activeTab === 'scan' && (
            <div className="space-y-4 animate-fadeIn">
              
              <div className="bg-white border-[3px] border-black rounded-2xl p-1.5 shadow-[3px_3px_0px_#121212] flex gap-1">
                {[
                  { id: 'pay', name: 'Palm Pay Auth', desc: 'Payment Token', icon: '💳' },
                  { id: 'door', name: 'Door Access', desc: 'Secure Entry', icon: '🔑' },
                  { id: 'vault', name: 'Identity Verify', desc: 'High Security', icon: '🛡️' },
                ].map(act => {
                  const isSel = selectedAuthAction.id === act.id;
                  return (
                    <button
                      key={act.id}
                      onClick={() => setSelectedAuthAction(act)}
                      className={`flex-1 py-2 px-1 rounded-xl text-xs font-black transition-all neo-btn flex flex-col items-center ${
                        isSel ? 'bg-[#FFDE59] border-[2px] border-black shadow-[2px_2px_0px_#121212]' : 'text-[#666]'
                      }`}
                    >
                      <span className="text-base">{act.icon}</span>
                      <span className="text-[10px] truncate max-w-[90px] font-display">{act.name}</span>
                      <span className="text-[8px] text-[#444]">{act.desc}</span>
                    </button>
                  );
                })}
              </div>

              {/* Live Camera Viewport with Countdown Overlay */}
              <div className="bg-white border-[3px] border-black rounded-3xl p-4 shadow-[6px_6px_0px_#121212] relative overflow-hidden flex flex-col items-center">
                <span className="absolute top-2.5 left-2.5 text-xs font-black text-black select-none">+</span>
                <span className="absolute top-2.5 right-2.5 text-xs font-black text-black select-none">+</span>

                {/* Live Stream Frame */}
                <div className="relative w-52 h-48 rounded-2xl border-[3px] border-black shadow-[4px_4px_0px_#121212] overflow-hidden bg-black flex items-center justify-center my-1.5">
                  {cameraReady ? (
                    <img 
                      src="/api/video_feed" 
                      alt="Live Camera Feed" 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-center p-3 text-white">
                      <PalmIcon className="w-12 h-12 text-[#FFDE59] mx-auto mb-1" />
                      <span className="text-[10px] font-black">CONNECT PI CAMERA</span>
                    </div>
                  )}

                  {/* On-Screen 3s Countdown Overlay */}
                  {scanCountdown !== null && (
                    <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center animate-fadeIn">
                      <span className="font-display font-black text-6xl text-[#FFDE59] drop-shadow-[2px_2px_0px_#000] animate-bounce">
                        {scanCountdown}
                      </span>
                      <span className="text-[11px] font-black text-white bg-black/80 px-2 py-0.5 rounded-md mt-1">
                        STEADY PALM
                      </span>
                    </div>
                  )}

                  {isScanning && (
                    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center animate-fadeIn text-white">
                      <RefreshCw className="w-8 h-8 animate-spin text-[#38BDF8] mb-1" />
                      <span className="text-xs font-black text-[#CCFF00]">AUTHENTICATING...</span>
                    </div>
                  )}
                </div>

                <div className="mt-1 text-center">
                  <span className={`inline-block px-3.5 py-1 rounded-full text-xs font-black border-[2px] border-black shadow-[2px_2px_0px_#121212] ${
                    scanCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : isScanning ? 'bg-[#FF7A00] text-white' : cameraReady ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                  }`}>
                    {scanCountdown !== null ? `STEADY PALM (${scanCountdown}s)...` : isScanning ? 'MATCHING GABOR VEINCODE...' : cameraReady ? `READY: ${selectedAuthAction.name.toUpperCase()}` : 'CAMERA OFFLINE'}
                  </span>
                </div>
              </div>

              {/* Action Button */}
              <div className="space-y-2">
                <button
                  onClick={() => handleScanWithCountdown(selectedAuthAction.name)}
                  disabled={isScanning || scanCountdown !== null || !cameraReady}
                  className="w-full py-4 bg-[#FFDE59] border-[3px] border-black rounded-2xl shadow-[5px_5px_0px_#121212] font-display font-black text-lg flex items-center justify-center gap-3 neo-btn hover:bg-[#ffe26b] disabled:opacity-50"
                >
                  {scanCountdown !== null ? (
                    <>
                      <Timer className="w-6 h-6 animate-spin" />
                      <span>HOLD STEADY: {scanCountdown}s...</span>
                    </>
                  ) : isScanning ? (
                    <>
                      <RefreshCw className="w-6 h-6 animate-spin" />
                      <span>READING CAMERA SENSOR...</span>
                    </>
                  ) : (
                    <>
                      <span>START 3s PALM SCAN</span>
                      <ArrowRight className="w-6 h-6 stroke-[3]" />
                    </>
                  )}
                </button>
              </div>

              {/* Last Scan Result Card */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-4 shadow-[4px_4px_0px_#121212]">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-black uppercase text-[#888] tracking-wider">RECENT VERIFICATION</span>
                  <Clock className="w-3.5 h-3.5 text-[#888]" />
                </div>
                {lastScan ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {lastScan.clahe_base64 ? (
                        <img 
                          src={`data:image/png;base64,${lastScan.clahe_base64}`} 
                          alt="Vein ROI" 
                          className="w-11 h-11 rounded-xl border-[2px] border-black shadow-[2px_2px_0px_#121212] object-cover bg-black"
                        />
                      ) : (
                        <div className={`w-10 h-10 rounded-xl border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black ${
                          lastScan.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                        }`}>
                          {lastScan.accepted ? '✓' : '✕'}
                        </div>
                      )}
                      <div>
                        <h4 className="font-display font-black text-sm">{lastScan.username || 'Unrecognized Palm'}</h4>
                        <p className="text-xs font-bold text-[#666]">Score: {lastScan.score.toFixed(4)} ({lastScan.time_ms}ms)</p>
                      </div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-lg border-[2px] border-black text-xs font-black shadow-[2px_2px_0px_#121212] ${
                      lastScan.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                    }`}>
                      {lastScan.accepted ? 'VERIFIED' : 'FAILED'}
                    </span>
                  </div>
                ) : (
                  <p className="text-xs font-bold text-[#888] italic">No authorizations recorded yet.</p>
                )}
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════
              PAGE 3: 6-SAMPLE ENROLLMENT STUDIO WITH 5s COUNTDOWN
             ══════════════════════════════════════════════════════════ */}
          {activeTab === 'enroll' && (
            <div className="space-y-4 animate-fadeIn">
              <div>
                <h2 className="font-display font-black text-xl tracking-tight">ENROLL PALM TEMPLATES</h2>
                <p className="text-xs font-bold text-[#666]">Automatic 5-second timer per capture for hand adjustment</p>
              </div>

              {/* Username Input Card */}
              <div className="space-y-1.5">
                <label className="text-xs font-black uppercase tracking-wider text-black">USERNAME / USER ID</label>
                <div className="relative">
                  <input
                    type="text"
                    value={enrollUsername}
                    onChange={e => setEnrollUsername(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                    placeholder="e.g. yesh-palm"
                    className="w-full px-4 py-3 bg-white border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-sm outline-none focus:bg-[#FFFDF0]"
                  />
                  <div className="absolute right-3 top-2.5 text-xs font-black px-2 py-0.5 bg-[#FFDE59] border-[1.5px] border-black rounded-md">
                    ID
                  </div>
                </div>
              </div>

              {/* 6 Step Indicators */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-4 shadow-[4px_4px_0px_#121212] space-y-2.5">
                <div className="flex justify-between items-center">
                  <span className="font-display font-black text-xs uppercase tracking-wider">LIVE SAMPLE PROGRESS</span>
                  <span className="text-xs font-black px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded-full">
                    {enrollSamples.length} / 6 SAMPLES
                  </span>
                </div>

                <div className="grid grid-cols-6 gap-2">
                  {[0, 1, 2, 3, 4, 5].map(idx => {
                    const sample = enrollSamples[idx];
                    const isDone = !!sample;
                    return (
                      <div
                        key={idx}
                        className={`h-12 rounded-xl border-[2.5px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black text-sm transition-all neo-btn overflow-hidden ${
                          isDone ? 'bg-[#CCFF00] scale-105' : 'bg-[#F4F4F0] text-[#888]'
                        }`}
                      >
                        {isDone && sample.thumb ? (
                          <img src={`data:image/png;base64,${sample.thumb}`} alt={`Sample ${idx+1}`} className="w-full h-full object-cover" />
                        ) : isDone ? (
                          '✓'
                        ) : (
                          idx + 1
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Live Camera Viewport — shown during enrollment */}
              <div className="bg-white border-[3px] border-black rounded-3xl p-4 shadow-[6px_6px_0px_#121212] relative overflow-hidden flex flex-col items-center">
                <div className="relative w-52 h-48 rounded-2xl border-[3px] border-black shadow-[4px_4px_0px_#121212] overflow-hidden bg-black flex items-center justify-center my-1.5">
                  {cameraReady ? (
                    <img
                      src="/api/video_feed"
                      alt="Live Camera Feed"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-center p-3 text-white">
                      <PalmIcon className="w-12 h-12 text-[#FFDE59] mx-auto mb-1" />
                      <span className="text-[10px] font-black">CONNECT PI CAMERA</span>
                    </div>
                  )}

                  {/* Countdown overlay during 5s timer */}
                  {enrollCountdown !== null && (
                    <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center animate-fadeIn">
                      <span className="font-display font-black text-6xl text-[#FFDE59] drop-shadow-[2px_2px_0px_#000] animate-bounce">
                        {enrollCountdown}
                      </span>
                      <span className="text-[11px] font-black text-white bg-black/80 px-2 py-0.5 rounded-md mt-1">
                        HOLD STEADY
                      </span>
                    </div>
                  )}

                  {isCapturingSample && (
                    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center animate-fadeIn text-white">
                      <RefreshCw className="w-8 h-8 animate-spin text-[#38BDF8] mb-1" />
                      <span className="text-xs font-black text-[#CCFF00]">CAPTURING...</span>
                    </div>
                  )}
                </div>
              </div>

              {/* 5-Second Timer Action Button */}
              <div className="space-y-2.5">
                <button
                  onClick={handleCaptureSampleWithCountdown}
                  disabled={isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !enrollUsername.trim() || !cameraReady}
                  className={`w-full py-4 border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-base flex items-center justify-center gap-2 neo-btn disabled:opacity-50 ${
                    enrollCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : 'bg-[#FF4081] text-white hover:bg-[#ff2872]'
                  }`}
                >
                  {enrollCountdown !== null ? (
                    <>
                      <Timer className="w-6 h-6 animate-spin" />
                      <span>POSITION HAND: CAPTURING IN {enrollCountdown}s...</span>
                    </>
                  ) : isCapturingSample ? (
                    <>
                      <RefreshCw className="w-6 h-6 animate-spin" />
                      <span>PROCESSING VEINCODE...</span>
                    </>
                  ) : enrollSamples.length >= 6 ? (
                    <span>ALL 6 SAMPLES COLLECTED ✓</span>
                  ) : (
                    <>
                      <Camera className="w-5 h-5" />
                      <span>START 5s TIMER FOR SAMPLE [{enrollSamples.length + 1}/6]</span>
                    </>
                  )}
                </button>

                <button
                  onClick={handleSaveEnrollment}
                  disabled={enrollSamples.length < 3 || !enrollUsername.trim()}
                  className="w-full py-3.5 bg-[#FFDE59] text-black border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-sm flex items-center justify-center gap-2 neo-btn hover:bg-[#ffe26b] disabled:opacity-40"
                >
                  <CheckCircle2 className="w-5 h-5" />
                  <span>SAVE TO BIOMETRIC DATABASE ({enrollSamples.length} SAMPLES)</span>
                </button>
              </div>

              {/* Positioning Guidance Banner */}
              <div className={`border-[3px] border-black rounded-2xl p-3.5 shadow-[4px_4px_0px_#121212] ${
                enrollCountdown !== null ? 'bg-[#FFDE59] text-black' : enrollStatusMsg.includes('failed') || enrollStatusMsg.includes('not detected') ? 'bg-[#FF7A00] text-white' : 'bg-[#38BDF8] text-black'
              }`}>
                <div className="flex items-center gap-1.5 mb-1">
                  <Timer className="w-4 h-4" />
                  <h4 className="font-display font-black text-xs uppercase">
                    {enrollCountdown !== null ? `TIMED POSITIONING (${enrollCountdown}s)` : 'POSITIONING INSTRUCTION'}
                  </h4>
                </div>
                <p className="text-xs font-bold leading-snug">
                  {enrollStatusMsg || SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm flat ~10–15cm above sensor. Tap the button to start the 5-second capture timer.'}
                </p>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════
              PAGE 4: FRIENDS & USER PROFILES
             ══════════════════════════════════════════════════════════ */}
          {activeTab === 'users' && (
            <div className="space-y-3.5 animate-fadeIn">
              
              <div className="relative">
                <Search className="absolute left-3.5 top-3.5 w-4 h-4 text-black stroke-[3]" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Search enrolled users..."
                  className="w-full pl-10 pr-12 py-2.5 bg-white border-[3px] border-black rounded-2xl shadow-[3px_3px_0px_#121212] font-display font-bold text-sm outline-none"
                />
                <button 
                  onClick={() => setActiveTab('enroll')}
                  className="absolute right-1.5 top-1.5 w-8 h-8 bg-[#FFDE59] border-[2px] border-black rounded-full shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black neo-btn"
                >
                  <Plus className="w-4 h-4 stroke-[3]" />
                </button>
              </div>

              <div className="space-y-2.5">
                {filteredUsers.length > 0 ? (
                  filteredUsers.map((u, i) => {
                    const avatarColor = ['bg-[#FFDE59]', 'bg-[#38BDF8]', 'bg-[#CCFF00]', 'bg-[#FF4081]', 'bg-[#A855F7]'][i % 5];
                    return (
                      <div
                        key={u.username}
                        className="bg-white border-[3px] border-black rounded-2xl p-3 shadow-[3px_3px_0px_#121212] flex items-center justify-between neo-card neo-card-hover"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-11 h-11 rounded-full ${avatarColor} border-[2.5px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black text-lg`}>
                            👤
                          </div>
                          <div>
                            <h4 className="font-display font-black text-sm leading-tight">{u.username}</h4>
                            <div className="flex items-center gap-1 mt-0.5">
                              <span className="text-[10px] font-black text-[#00aa44]">Active Template</span>
                              <span className="text-[9px] text-[#888] ml-1">• {u.sample_count} Samples</span>
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={() => setDeleteTarget(u.username)}
                          className="w-8 h-8 rounded-xl bg-[#FF4081] text-white border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center neo-btn hover:bg-[#ff2872]"
                          title="Delete profile"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    );
                  })
                ) : (
                  <div className="bg-[#FFDE59] border-[3px] border-black rounded-2xl p-6 shadow-[4px_4px_0px_#121212] text-center space-y-2">
                    <p className="font-display font-black text-base">No Enrolled Users</p>
                    <p className="text-xs font-bold text-[#333]">Tap the ENROLL tab to capture live palm templates.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════
              PAGE 5: SYSTEM & ACCURACY DIAGNOSTICS
             ══════════════════════════════════════════════════════════ */}
          {activeTab === 'admin' && (
            <div className="space-y-3.5 animate-fadeIn">
              <div>
                <h2 className="font-display font-black text-xl tracking-tight">SYSTEM DIAGNOSTICS</h2>
                <p className="text-xs font-bold text-[#666]">Hardware status &amp; biometric separation matrix</p>
              </div>

              <div 
                onClick={openReport}
                className="bg-[#FFDE59] border-[3px] border-black rounded-2xl p-4 shadow-[4px_4px_0px_#121212] cursor-pointer neo-btn flex items-center justify-between"
              >
                <div>
                  <h3 className="font-display font-black text-sm">ACCURACY REPORT MATRIX</h3>
                  <p className="text-xs font-bold text-[#444]">View Self-Match &amp; Cross-Match scores</p>
                </div>
                <div className="w-9 h-9 rounded-xl bg-white border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black">
                  📊
                </div>
              </div>

              <div className="bg-white border-[3px] border-black rounded-2xl p-4 shadow-[4px_4px_0px_#121212] space-y-2">
                <h4 className="font-display font-black text-xs uppercase tracking-wider text-[#888]">COMPUTE SPECIFICATIONS</h4>
                <div className="grid grid-cols-2 gap-2 text-xs font-black">
                  <div className="p-2.5 bg-[#FFFDF0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212]">
                    <span className="block text-[10px] text-[#666]">CAMERA DRIVER</span>
                    <span>{cameraType}</span>
                  </div>
                  <div className="p-2.5 bg-[#FFFDF0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212]">
                    <span className="block text-[10px] text-[#666]">PARALLEL MATCHER</span>
                    <span>4-Core Worker Pool</span>
                  </div>
                  <div className="p-2.5 bg-[#FFFDF0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212]">
                    <span className="block text-[10px] text-[#666]">LAYER 1 RAM FILTER</span>
                    <span>16-Float Euclidean</span>
                  </div>
                  <div className="p-2.5 bg-[#FFFDF0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212]">
                    <span className="block text-[10px] text-[#666]">THRESHOLD</span>
                    <span>MNHD &le; 0.3800</span>
                  </div>
                </div>
              </div>

              <div className="bg-[#CCFF00] border-[3px] border-black rounded-2xl p-3.5 shadow-[4px_4px_0px_#121212] flex justify-between items-center">
                <div>
                  <h4 className="font-display font-black text-sm uppercase">SQLite Storage Vault</h4>
                  <p className="text-xs font-bold text-black">
                    {users.length} Users • {users.reduce((acc, u) => acc + u.sample_count, 0)} Templates stored (zlib)
                  </p>
                </div>
                <Database className="w-6 h-6 text-black" />
              </div>

              <button
                onClick={async () => {
                  if (!window.confirm('DELETE ALL enrolled users and templates? This cannot be undone.')) return;
                  const res = await fetch('/api/database/reset', { method: 'DELETE' });
                  if (res.ok) {
                    showToast('Database cleared. Re-enroll using live camera.', 'warn');
                    loadUsers();
                  }
                }}
                className="w-full py-3 bg-[#FF4081] text-white border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-sm neo-btn"
              >
                🗑️ RESET DATABASE (Clear All Users)
              </button>
            </div>
          )}
        </main>

        {/* ── SMOOTH SLIDING NEOBRUTALISM BOTTOM NAVBAR ── */}
        <nav className="absolute bottom-0 left-0 right-0 h-[76px] bg-[#FFFDF0] border-t-[3px] border-black px-2 flex items-center z-20">
          <div className="relative w-full h-[52px] flex items-center">
            
            {/* Sliding Highlight Pill with Bounce Spring */}
            <div 
              className="absolute top-0 bottom-0 bg-[#FFDE59] border-[2.5px] border-black rounded-full shadow-[2.5px_2.5px_0px_#121212] transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] pointer-events-none"
              style={{
                width: '18.4%',
                left: `calc(${activeTabIdx * 20}% + 0.8%)`,
              }}
            />

            {/* Nav Tabs */}
            {tabList.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative z-10 flex-1 h-full flex flex-col items-center justify-center transition-all group neo-btn`}
                >
                  <Icon className={`w-4 h-4 stroke-[2.5] transition-transform duration-200 ${
                    isActive ? 'scale-110 text-black' : 'text-[#666] group-hover:text-black group-hover:scale-105'
                  }`} />
                  <span className={`text-[10px] font-display font-black tracking-tight mt-0.5 transition-colors ${
                    isActive ? 'text-black font-black' : 'text-[#666] group-hover:text-black'
                  }`}>
                    {tab.label}
                  </span>
                </button>
              );
            })}

          </div>
        </nav>

        {/* ── FLOATING ACTION (+) BUTTON ON USERS VIEW ── */}
        {activeTab === 'users' && (
          <button
            onClick={() => setActiveTab('enroll')}
            className="absolute bottom-24 right-5 w-14 h-14 bg-[#FFDE59] border-[3px] border-black rounded-full shadow-[4px_4px_0px_#121212] flex items-center justify-center font-black text-2xl neo-btn z-30"
          >
            <Plus className="w-7 h-7 stroke-[3]" />
          </button>
        )}

        {/* ── FULLSCREEN RESULT OVERLAY ── */}
        {resultOverlay && (
          <div className={`absolute inset-0 z-50 p-6 flex flex-col items-center justify-center animate-fadeIn ${
            resultOverlay.accepted ? 'bg-[#38BDF8]' : 'bg-[#FF4081]'
          }`}>
            <div className="w-full bg-white border-[4px] border-black rounded-3xl p-6 shadow-[8px_8px_0px_#121212] text-center space-y-4">
              <div className={`w-20 h-20 mx-auto rounded-full border-[3px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center font-display font-black text-3xl ${
                resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
              }`}>
                {resultOverlay.accepted ? '✓' : '✕'}
              </div>

              <div>
                <h3 className="font-display font-black text-2xl tracking-tight uppercase">
                  {resultOverlay.accepted ? (resultOverlay.action_type || 'AUTHENTICATED') : 'NOT RECOGNISED'}
                </h3>
                <p className="font-bold text-sm text-[#444] mt-1">
                  {resultOverlay.accepted 
                    ? `Biometric token confirmed for ${resultOverlay.username}`
                    : 'Palm does not match enrolled records in database'}
                </p>
              </div>

              {/* Confidence Gauge */}
              <div className="bg-[#FFFDF0] border-[2px] border-black rounded-xl p-3 shadow-[2px_2px_0px_#121212] space-y-1 text-left">
                <div className="flex justify-between text-xs font-black">
                  <span>MNHD SCORE</span>
                  <span>{resultOverlay.score.toFixed(4)}</span>
                </div>
                <div className="w-full h-3.5 bg-[#E2E8F0] border-[1.5px] border-black rounded-full overflow-hidden">
                  <div
                    className={`h-full ${resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081]'}`}
                    style={{ width: `${Math.max(10, Math.min(100, (1 - resultOverlay.score / 0.5) * 100))}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] font-bold text-[#888] pt-0.5">
                  <span>Threshold: &lt; 0.3800</span>
                  <span>Latency: {resultOverlay.time_ms}ms</span>
                </div>
              </div>

              <button
                onClick={() => setResultOverlay(null)}
                className="w-full py-3 bg-[#FFDE59] border-[2.5px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm neo-btn"
              >
                DONE
              </button>
            </div>
          </div>
        )}

        {/* ── DELETE CONFIRMATION MODAL ── */}
        {deleteTarget && (
          <div className="absolute inset-0 bg-black/60 z-50 flex items-center justify-center p-6 animate-fadeIn">
            <div className="w-full bg-white border-[4px] border-black rounded-3xl p-6 shadow-[6px_6px_0px_#121212] text-center space-y-4">
              <h3 className="font-display font-black text-xl">DELETE PROFILE?</h3>
              <p className="text-sm font-bold text-[#666]">
                Permanently delete biometric template for <span className="text-[#FF4081] font-black">'{deleteTarget}'</span>?
              </p>
              <div className="grid grid-cols-2 gap-3 pt-2">
                <button
                  onClick={() => setDeleteTarget(null)}
                  className="py-3 bg-[#F4F4F0] border-[2.5px] border-black rounded-xl shadow-[2px_2px_0px_#121212] font-display font-black text-xs neo-btn"
                >
                  CANCEL
                </button>
                <button
                  onClick={confirmDelete}
                  className="py-3 bg-[#FF4081] text-white border-[2.5px] border-black rounded-xl shadow-[2px_2px_0px_#121212] font-display font-black text-xs neo-btn"
                >
                  DELETE
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── ACCURACY REPORT MODAL ── */}
        {reportModalOpen && (
          <div className="absolute inset-0 bg-black/60 z-50 flex items-center justify-center p-5 animate-fadeIn">
            <div className="w-full max-h-[90%] bg-white border-[4px] border-black rounded-3xl p-5 shadow-[6px_6px_0px_#121212] flex flex-col space-y-3 overflow-hidden">
              <div className="flex justify-between items-center border-b-2 border-black pb-2">
                <h3 className="font-display font-black text-base">BIOMETRIC MATRIX REPORT</h3>
                <button onClick={() => setReportModalOpen(false)} className="font-black text-lg px-2">✕</button>
              </div>

              <div className="overflow-y-auto space-y-3 flex-1 text-xs font-bold pr-1">
                <div>
                  <h4 className="font-display font-black text-xs uppercase mb-1">Self-Match (Intra-User)</h4>
                  {reportData?.self_matches?.length ? (
                    <div className="space-y-1">
                      {reportData.self_matches.map(([u, mn, av, mx, q]) => (
                        <div key={u} className="p-2 bg-[#FFFDF0] border border-black rounded-lg flex justify-between">
                          <span>{u}</span>
                          <span>min:{mn.toFixed(3)} avg:{av.toFixed(3)} [{q}]</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[#888] italic">Need &ge; 2 samples to compute self-match.</p>
                  )}
                </div>

                <div>
                  <h4 className="font-display font-black text-xs uppercase mb-1">Cross-Match Separation</h4>
                  {reportData?.cross_matches?.length ? (
                    <div className="space-y-1">
                      {reportData.cross_matches.map(([pair, sc, stat]) => (
                        <div key={pair} className="p-2 bg-[#FFFDF0] border border-black rounded-lg flex justify-between">
                          <span>{pair}</span>
                          <span className={stat === 'OK' ? 'text-[#00aa44]' : 'text-[#FF4081]'}>{sc.toFixed(4)} [{stat}]</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[#888] italic">Need &ge; 2 users to compute cross-match.</p>
                  )}
                </div>
              </div>

              <button
                onClick={() => setReportModalOpen(false)}
                className="w-full py-2.5 bg-[#FFDE59] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212] font-display font-black text-xs neo-btn"
              >
                CLOSE
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
