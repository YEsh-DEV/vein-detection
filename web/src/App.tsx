import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import { 
  CheckCircle2, 
  XCircle, 
  Camera, 
  RefreshCw, 
  Clock, 
  ArrowRight, 
  ArrowLeft,
  Timer, 
  AlertTriangle,
  UserPlus,
  X,
  ShieldCheck,
  Sparkles,
  Database,
  Lock
} from 'lucide-react';

type AppState = 'idle' | 'scan' | 'enroll';

interface ScanResult {
  accepted: boolean;
  username: string | null;
  score: number;
  threshold: number;
  time_ms: number;
  clahe_base64?: string;
}

interface ReportData {
  self_matches?: Array<{ username: string; min_score: number; avg_score: number; max_score: number; quality: string }>;
  cross_matches?: Array<{ pair: string; score: number; status: string }>;
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
  // 3-Screen State Machine
  const [appState, setAppState] = useState<AppState>('idle');

  // Hardware Status
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraType, setCameraType] = useState('Checking...');
  const [totalUsers, setTotalUsers] = useState(0);

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

  // Hidden Admin / Ops Modal
  const [adminOpen, setAdminOpen] = useState(false);
  const [adminTapCount, setAdminTapCount] = useState(0);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  // Video settled state for Idle Screen
  const [videoSettled, setVideoSettled] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Active state ref to cancel async countdowns on navigation
  const appStateRef = useRef<AppState>(appState);
  useEffect(() => {
    appStateRef.current = appState;
  }, [appState]);

  // Toast Notification
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'warn' | 'error' } | null>(null);

  const showToast = (msg: string, type: 'success' | 'warn' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  // Poll Hardware Status
  const loadStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setCameraReady(data.camera_available ?? false);
        setCameraType(data.camera_type || 'None');
        setTotalUsers(data.users_count || 0);
      }
    } catch {
      setCameraReady(false);
      setCameraType('Disconnected');
    }
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Cancel enrollment when leaving enroll state with partial samples
  useEffect(() => {
    if (appState !== 'enroll' && enrollSamples.length > 0 && enrollUsername) {
      fetch('/api/enroll/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: enrollUsername }),
      }).catch(() => {});
      setEnrollSamples([]);
      setEnrollUsername('');
      setEnrollStatusMsg('');
    }
  }, [appState]);

  // Idle Timeout: if Scan screen has no interaction for 30s, auto-return to Idle
  useEffect(() => {
    if (appState !== 'scan') return;
    if (isScanning || scanCountdown !== null || resultOverlay !== null) return;

    const timeout = setTimeout(() => {
      setAppState('idle');
    }, 30000);

    return () => clearTimeout(timeout);
  }, [appState, isScanning, scanCountdown, resultOverlay]);

  // Scan Execution with 3-Second Countdown
  const handleScanWithCountdown = async () => {
    if (isScanning || scanCountdown !== null) return;
    
    // 3-Second countdown with state abort check
    for (let i = 3; i > 0; i--) {
      if (appStateRef.current !== 'scan') {
        setScanCountdown(null);
        return;
      }
      setScanCountdown(i);
      await new Promise(r => setTimeout(r, 1000));
    }
    if (appStateRef.current !== 'scan') {
      setScanCountdown(null);
      return;
    }
    setScanCountdown(null);
    setIsScanning(true);

    try {
      const res = await fetch('/api/scan', { method: 'POST' });
      if (res.ok) {
        const data: ScanResult = await res.json();
        setLastScan(data);
        setResultOverlay(data);

        if (data.accepted) {
          confetti({
            particleCount: 90,
            spread: 75,
            origin: { y: 0.6 },
            colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00', '#121212']
          });
        }

        // Auto-return to Idle state after 4.5 seconds
        setTimeout(() => {
          setResultOverlay(null);
          setAppState('idle');
        }, 4500);

      } else {
        const err = await res.json();
        showToast(err.detail || 'Scan failed: Palm not detected', 'warn');
        // Auto-return to idle after failed attempt after delay
        setTimeout(() => {
          setAppState('idle');
        }, 4000);
      }
    } catch {
      showToast('Cannot connect to server. Ensure server.py is running.', 'error');
    } finally {
      setIsScanning(false);
    }
  };

  // Live Camera Sample Capture with 5-Second Countdown
  const handleCaptureSampleWithCountdown = async () => {
    if (isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6) return;
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (!cleanUname) {
      showToast('Enter a username first!', 'warn');
      return;
    }

    const currentHint = SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm steady ~10-15cm above sensor';
    setEnrollStatusMsg(`${currentHint} (Capturing in 5 seconds...)`);

    // 5-Second Countdown with state abort check
    for (let i = 5; i > 0; i--) {
      if (appStateRef.current !== 'enroll') {
        setEnrollCountdown(null);
        return;
      }
      setEnrollCountdown(i);
      await new Promise(r => setTimeout(r, 1000));
    }
    if (appStateRef.current !== 'enroll') {
      setEnrollCountdown(null);
      return;
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
        showToast(`Sample ${enrollSamples.length + 1}/6 captured!`, 'success');
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

  // Commit Enrollment to Database
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
        showToast(`ENROLLED '${cleanUname}' successfully!`, 'success');
        confetti({
          particleCount: 100,
          spread: 80,
          origin: { y: 0.5 },
          colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00']
        });
        setEnrollUsername('');
        setEnrollSamples([]);
        setEnrollStatusMsg('');
        loadStatus();
        // Explicit transition: return to Idle
        setTimeout(() => setAppState('idle'), 1500);
      } else {
        const err = await res.json();
        showToast(err.detail || 'Failed to save enrollment to database.', 'error');
      }
    } catch {
      showToast('Network error saving to database.', 'error');
    }
  };

  // Hidden Admin: Tap camera bead 5 times
  const handleSecretAdminTap = () => {
    const next = adminTapCount + 1;
    if (next >= 5) {
      setAdminTapCount(0);
      setAdminOpen(true);
      fetchReport();
    } else {
      setAdminTapCount(next);
      setTimeout(() => setAdminTapCount(0), 3000);
    }
  };

  const fetchReport = async () => {
    setReportLoading(true);
    try {
      const res = await fetch('/api/report');
      if (res.ok) {
        const data = await res.json();
        setReportData(data);
      }
    } catch {
      setReportData(null);
    } finally {
      setReportLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    if (!window.confirm('WARNING: DELETE ALL enrolled users and biometric templates?')) return;
    try {
      const res = await fetch('/api/database/reset', { method: 'DELETE' });
      if (res.ok) {
        showToast('Database wiped. System reset.', 'warn');
        loadStatus();
        fetchReport();
      }
    } catch {
      showToast('Error resetting database.', 'error');
    }
  };

  return (
    <div className="min-h-screen bg-dribbble-yellow flex justify-center items-center p-0 sm:p-6 text-[#121212] select-none font-sans">
      
      {/* ── KIOSK PHONE / 5" TOUCHSCREEN CONTAINER (480px) ── */}
      <div className="w-full max-w-[480px] min-h-screen sm:min-h-[854px] sm:h-[854px] bg-[#FFFDF0] border-x-0 sm:border-[4px] border-black sm:rounded-[36px] sm:shadow-[10px_10px_0px_#121212] flex flex-col relative overflow-hidden bg-neo-cream">

        {/* ── MINIMAL SERVICE STATUS BAR ── */}
        <div className="px-5 pt-3 pb-2 flex items-center justify-between text-xs font-black text-black z-20 border-b-[2px] border-black/10">
          <div className="flex items-center gap-2">
            {/* Secret 5-tap Admin trigger on camera bead */}
            <button 
              onClick={handleSecretAdminTap}
              className="flex items-center gap-1.5 focus:outline-none"
              title="Camera status"
            >
              <span className={`w-3 h-3 rounded-full border-[1.5px] border-black transition-colors ${cameraReady ? 'bg-[#CCFF00]' : 'bg-[#FF4081]'}`} />
              <span className="text-[10px] uppercase font-bold tracking-tight text-[#444]">{cameraType}</span>
            </button>
          </div>

          <div className="text-center font-display font-black text-xs tracking-wider uppercase">
            PALM PAY TERMINAL
          </div>

          <div className="text-[11px] font-mono font-black text-[#555]">
            {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>

        {/* ── TOAST ALERT BANNER ── */}
        {toast && (
          <div className="absolute top-14 left-5 right-5 z-50 animate-bounce">
            <div className={`p-3 border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-xs text-center flex items-center justify-center gap-2 ${
              toast.type === 'error' ? 'bg-[#FF4081] text-white' : toast.type === 'warn' ? 'bg-[#FF7A00] text-white' : 'bg-[#CCFF00] text-black'
            }`}>
              {toast.type === 'error' || toast.type === 'warn' ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
              <span>{toast.msg}</span>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            SCREEN 1: IDLE / HERO SCREEN
            - Plays intro animation/video on wake/boot
            - Settles into tactile "ready" card
            - ANY touch/click skips immediately to Scan screen
           ══════════════════════════════════════════════════════════════════════ */}
        {appState === 'idle' && (
          <div 
            onClick={() => setAppState('scan')}
            className="flex-1 flex flex-col items-center justify-between p-6 cursor-pointer animate-fadeIn relative overflow-hidden select-none"
          >
            {/* Optional Background Intro Video (provided separately as /intro.mp4) */}
            <video
              ref={videoRef}
              src="/intro.mp4"
              autoPlay
              muted
              playsInline
              onEnded={() => setVideoSettled(true)}
              onError={() => setVideoSettled(true)}
              className={`absolute inset-0 w-full h-full object-cover z-0 pointer-events-none transition-opacity duration-700 ${videoSettled ? 'opacity-0' : 'opacity-100'}`}
            />

            {/* Top Badge */}
            <div className="w-full flex justify-between items-center z-10">
              <div className="px-3 py-1 bg-[#FFDE59] border-[2px] border-black rounded-xl text-[11px] font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 fill-black" />
                <span>SUB-DERMAL NIR 850nm</span>
              </div>
              <div className="px-2.5 py-1 bg-[#38BDF8] border-[2px] border-black rounded-xl text-[10px] font-black shadow-[2px_2px_0px_#121212]">
                {totalUsers} ENROLLED
              </div>
            </div>

            {/* Main Center "Ready" Card */}
            <div className="w-full bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-6 shadow-[8px_8px_0px_#121212] flex flex-col items-center text-center my-auto z-10 neo-btn">
              
              {/* Animated Vein Sensor Graphic */}
              <div className="relative my-4">
                <div className="w-28 h-28 rounded-3xl bg-[#CCFF00] border-[3.5px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center animate-float">
                  <PalmIcon className="w-16 h-16 text-black" animated={true} />
                </div>
                <div className="absolute -bottom-2 -right-2 px-2.5 py-1 bg-[#FF4081] text-white border-[2px] border-black rounded-lg text-[9px] font-black shadow-[2px_2px_0px_#121212] animate-bounce">
                  ⚡ INSTANT
                </div>
              </div>

              <div className="space-y-2 mt-2">
                <h1 className="font-display font-black text-2xl leading-tight uppercase tracking-tight">
                  TAP OR PRESENT PALM<br />TO BEGIN
                </h1>
                <p className="text-xs font-bold text-[#666] max-w-[280px] mx-auto">
                  Contactless payment authorization powered by sub-dermal vascular recognition.
                </p>
              </div>

              {/* High-visibility Action Prompt */}
              <div className="mt-6 w-full py-4 bg-[#FFDE59] border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-base flex items-center justify-center gap-2 animate-pulse">
                <span>TOUCH SCREEN TO PAY</span>
                <ArrowRight className="w-5 h-5 stroke-[3]" />
              </div>
            </div>

            {/* Bottom Instruction */}
            <div className="text-center z-10">
              <span className="text-[11px] font-black text-[#555] uppercase tracking-wider">
                Touch anywhere to wake terminal • Auto-timeout 30s
              </span>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            SCREEN 2: SCAN (DEFAULT HOME SCREEN AFTER IDLE)
            - Full-bleed live camera viewport (/api/video_feed)
            - Payment mode hardcoded to Palm Pay Auth (no mode selectors)
            - 3s countdown → capture → result overlay → auto-return to Idle (4-5s)
            - Small low-emphasis icon in corner for Enroll
           ══════════════════════════════════════════════════════════════════════ */}
        {appState === 'scan' && (
          <div className="flex-1 flex flex-col p-5 space-y-4 animate-fadeIn overflow-y-auto">
            
            {/* Minimal Header Bar: Back to Idle on left, Enroll shortcut on right */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => setAppState('idle')}
                className="px-3 py-1.5 bg-white border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1 neo-btn"
              >
                <ArrowLeft className="w-3.5 h-3.5 stroke-[3]" />
                <span>Cancel</span>
              </button>

              <div className="px-3 py-1 bg-[#FFDE59] border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>PALM PAY AUTH</span>
              </div>

              {/* Small, low-emphasis corner trigger to access Enrollment */}
              <button
                onClick={() => setAppState('enroll')}
                className="w-8 h-8 bg-[#F4F4F0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212] flex items-center justify-center text-[#555] hover:text-black neo-btn"
                title="Enroll New Palm"
              >
                <UserPlus className="w-4 h-4" />
              </button>
            </div>

            {/* Dominant Full-Bleed Live Camera Viewport Box */}
            <div className="bg-white border-[3px] border-black rounded-3xl p-3 shadow-[6px_6px_0px_#121212] relative overflow-hidden flex flex-col items-center flex-1 justify-center">
              
              <div className="relative w-full h-full min-h-[340px] rounded-2xl border-[3px] border-black shadow-[4px_4px_0px_#121212] overflow-hidden bg-black flex items-center justify-center">
                {cameraReady ? (
                  <img 
                    src="/api/video_feed" 
                    alt="Live Camera Feed" 
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="text-center p-4 text-white">
                    <PalmIcon className="w-14 h-14 text-[#FFDE59] mx-auto mb-2" />
                    <span className="text-xs font-black uppercase">CAMERA OFFLINE</span>
                  </div>
                )}

                {/* 3-Second Countdown Overlay */}
                {scanCountdown !== null && (
                  <div className="absolute inset-0 bg-black/45 flex flex-col items-center justify-center animate-fadeIn">
                    <span className="font-display font-black text-7xl text-[#FFDE59] drop-shadow-[3px_3px_0px_#000] animate-bounce">
                      {scanCountdown}
                    </span>
                    <span className="text-xs font-black text-white bg-black/85 px-3 py-1 rounded-md mt-2 tracking-wider">
                      HOLD PALM STEADY
                    </span>
                  </div>
                )}

                {/* Processing Overlay */}
                {isScanning && (
                  <div className="absolute inset-0 bg-black/65 flex flex-col items-center justify-center animate-fadeIn text-white">
                    <RefreshCw className="w-10 h-10 animate-spin text-[#38BDF8] mb-2" />
                    <span className="text-sm font-black text-[#CCFF00] tracking-wider">AUTHENTICATING...</span>
                  </div>
                )}
              </div>

              {/* Status Bead Indicator */}
              <div className="mt-2.5 text-center">
                <span className={`inline-block px-4 py-1 rounded-full text-xs font-black border-[2px] border-black shadow-[2px_2px_0px_#121212] ${
                  scanCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : isScanning ? 'bg-[#FF7A00] text-white' : cameraReady ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                }`}>
                  {scanCountdown !== null ? `ALIGNING PALM (${scanCountdown}s)...` : isScanning ? 'MATCHING VEIN PATTERN...' : cameraReady ? 'READY FOR PALM AUTH' : 'CAMERA DISCONNECTED'}
                </span>
              </div>
            </div>

            {/* Primary Action Button */}
            <button
              onClick={handleScanWithCountdown}
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
                  <span>READING SENSOR...</span>
                </>
              ) : (
                <>
                  <span>START 3s PALM SCAN</span>
                  <ArrowRight className="w-6 h-6 stroke-[3]" />
                </>
              )}
            </button>

            {/* Last Scan Status Card */}
            {lastScan && (
              <div className="bg-white border-[3px] border-black rounded-2xl p-3 shadow-[3px_3px_0px_#121212] flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {lastScan.clahe_base64 ? (
                    <img 
                      src={`data:image/png;base64,${lastScan.clahe_base64}`} 
                      alt="ROI" 
                      className="w-10 h-10 rounded-xl border-[2px] border-black object-cover bg-black"
                    />
                  ) : (
                    <div className={`w-9 h-9 rounded-xl border-[2px] border-black flex items-center justify-center font-display font-black ${
                      lastScan.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                    }`}>
                      {lastScan.accepted ? '✓' : '✕'}
                    </div>
                  )}
                  <div>
                    <h4 className="font-display font-black text-xs">{lastScan.username || 'Unrecognized Palm'}</h4>
                    <p className="text-[10px] font-bold text-[#666]">Score: {lastScan.score.toFixed(4)} ({lastScan.time_ms}ms)</p>
                  </div>
                </div>

                <span className={`px-2.5 py-1 rounded-lg border-[2px] border-black text-[10px] font-black ${
                  lastScan.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                }`}>
                  {lastScan.accepted ? 'VERIFIED' : 'FAILED'}
                </span>
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            SCREEN 3: ENROLL (GUIDED 6-SAMPLE CAPTURE STUDIO)
            - Re-parented from tab navigation
            - Explicit Close/Back button returns directly to Idle
            - 6-sample guided capture with 5s countdown, thumbnails, posture guidance
            - On save: celebrates with confetti, returns to Idle
           ══════════════════════════════════════════════════════════════════════ */}
        {appState === 'enroll' && (
          <div className="flex-1 flex flex-col p-5 space-y-3.5 animate-fadeIn overflow-y-auto pb-6">
            
            {/* Header: Title + Explicit Close Button (Returns to Idle) */}
            <div className="flex items-center justify-between border-b-[2px] border-black/10 pb-2">
              <div>
                <h2 className="font-display font-black text-lg tracking-tight uppercase">ENROLL PALM TEMPLATES</h2>
                <p className="text-[11px] font-bold text-[#666]">Guided 6-sample multi-angle calibration</p>
              </div>

              <button
                onClick={() => setAppState('idle')}
                className="w-8 h-8 rounded-xl bg-[#F4F4F0] border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black text-sm neo-btn"
                title="Exit to Idle"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Username Input Field */}
            <div className="space-y-1">
              <label className="text-[11px] font-black uppercase tracking-wider text-black">USERNAME / ACCOUNT ID</label>
              <input
                type="text"
                value={enrollUsername}
                onChange={e => setEnrollUsername(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                placeholder="e.g. alex-palm"
                className="w-full px-3.5 py-2.5 bg-white border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm outline-none focus:bg-[#FFFDF0]"
              />
            </div>

            {/* 6-Cell Sample Matrix Grid */}
            <div className="bg-white border-[3px] border-black rounded-2xl p-3 shadow-[3px_3px_0px_#121212] space-y-2">
              <div className="flex justify-between items-center text-xs font-black">
                <span className="uppercase tracking-wider">PROGRESS:</span>
                <span className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded-full text-[10px]">
                  {enrollSamples.length} / 6 SAMPLES
                </span>
              </div>

              <div className="grid grid-cols-6 gap-1.5">
                {[0, 1, 2, 3, 4, 5].map(idx => {
                  const sample = enrollSamples[idx];
                  const isDone = !!sample;
                  return (
                    <div
                      key={idx}
                      className={`h-11 rounded-xl border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black text-xs transition-all overflow-hidden ${
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

            {/* Mini Camera Feed during Positioning */}
            <div className="bg-white border-[3px] border-black rounded-2xl p-3 shadow-[4px_4px_0px_#121212] relative overflow-hidden flex flex-col items-center">
              <div className="relative w-44 h-36 rounded-xl border-[2.5px] border-black overflow-hidden bg-black flex items-center justify-center">
                {cameraReady ? (
                  <img src="/api/video_feed" alt="Camera Feed" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-[10px] text-white font-black">CAMERA OFFLINE</span>
                )}

                {/* 5-Second Timer Overlay */}
                {enrollCountdown !== null && (
                  <div className="absolute inset-0 bg-black/45 flex flex-col items-center justify-center">
                    <span className="font-display font-black text-5xl text-[#FFDE59] drop-shadow-[2px_2px_0px_#000] animate-bounce">
                      {enrollCountdown}
                    </span>
                    <span className="text-[9px] font-black text-white bg-black/80 px-2 py-0.5 rounded mt-1">
                      HOLD STEADY
                    </span>
                  </div>
                )}

                {isCapturingSample && (
                  <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white">
                    <RefreshCw className="w-7 h-7 animate-spin text-[#38BDF8]" />
                  </div>
                )}
              </div>
            </div>

            {/* 5-Second Countdown Capture Button */}
            <button
              onClick={handleCaptureSampleWithCountdown}
              disabled={isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !enrollUsername.trim() || !cameraReady}
              className={`w-full py-3.5 border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm flex items-center justify-center gap-2 neo-btn disabled:opacity-50 ${
                enrollCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : 'bg-[#FF4081] text-white hover:bg-[#ff2872]'
              }`}
            >
              {enrollCountdown !== null ? (
                <>
                  <Timer className="w-5 h-5 animate-spin" />
                  <span>CAPTURING IN {enrollCountdown}s...</span>
                </>
              ) : isCapturingSample ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>PROCESSING VEINCODE...</span>
                </>
              ) : enrollSamples.length >= 6 ? (
                <span>ALL 6 SAMPLES COLLECTED ✓</span>
              ) : (
                <>
                  <Camera className="w-4 h-4" />
                  <span>START 5s TIMER FOR SAMPLE [{enrollSamples.length + 1}/6]</span>
                </>
              )}
            </button>

            {/* Save Enrollment Button (requires >= 3 samples) */}
            <button
              onClick={handleSaveEnrollment}
              disabled={enrollSamples.length < 3 || !enrollUsername.trim()}
              className="w-full py-3 bg-[#CCFF00] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs flex items-center justify-center gap-2 neo-btn hover:bg-[#b8e600] disabled:opacity-40"
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>SAVE TO DATABASE ({enrollSamples.length} SAMPLES)</span>
            </button>

            {/* Dynamic Posture Guidance Banner */}
            <div className={`border-[2.5px] border-black rounded-xl p-3 shadow-[3px_3px_0px_#121212] ${
              enrollCountdown !== null ? 'bg-[#FFDE59] text-black' : 'bg-[#38BDF8] text-black'
            }`}>
              <div className="flex items-center gap-1.5 mb-0.5">
                <Timer className="w-3.5 h-3.5" />
                <h4 className="font-display font-black text-[10px] uppercase">
                  {enrollCountdown !== null ? `AUTO-TIMED POSITIONING (${enrollCountdown}s)` : 'POSTURE GUIDANCE'}
                </h4>
              </div>
              <p className="text-[11px] font-bold leading-tight">
                {enrollStatusMsg || SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm flat ~10-15cm above sensor. Tap to begin 5-second countdown.'}
              </p>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            FULLSCREEN RESULT OVERLAY (FOR SCAN RESULTS)
            - Pops up on successful/failed authentication
            - Displays confidence gauge and user name
            - Auto-dismisses and returns to Idle after 4.5s
           ══════════════════════════════════════════════════════════════════════ */}
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
                  {resultOverlay.accepted ? 'PALM AUTH VERIFIED' : 'NOT RECOGNISED'}
                </h3>
                <p className="font-bold text-xs text-[#444] mt-1">
                  {resultOverlay.accepted 
                    ? `Payment token confirmed for '${resultOverlay.username}'`
                    : 'Palm vascular pattern not recognized in database'}
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
                  <span>Threshold: &le; 0.3800</span>
                  <span>Latency: {resultOverlay.time_ms}ms</span>
                </div>
              </div>

              <button
                onClick={() => {
                  setResultOverlay(null);
                  setAppState('idle');
                }}
                className="w-full py-3 bg-[#FFDE59] border-[2.5px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm neo-btn"
              >
                DONE (RETURN TO IDLE)
              </button>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            HIDDEN KIOSK ADMIN MODAL
            - Accessible only via 5 rapid taps on top-left camera indicator bead
            - Kept completely separate from the end-user touch terminal interface
            - Provides Reset Database & Accuracy Matrix diagnostics
           ══════════════════════════════════════════════════════════════════════ */}
        {adminOpen && (
          <div className="absolute inset-0 bg-black/75 z-50 flex items-center justify-center p-5 animate-fadeIn">
            <div className="w-full max-h-[92%] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-5 shadow-[8px_8px_0px_#121212] flex flex-col space-y-3 overflow-hidden text-xs">
              
              <div className="flex justify-between items-center border-b-2 border-black pb-2">
                <div className="flex items-center gap-2">
                  <Lock className="w-4 h-4 text-black" />
                  <h3 className="font-display font-black text-sm uppercase">KIOSK OPS ADMIN</h3>
                </div>
                <button 
                  onClick={() => setAdminOpen(false)} 
                  className="w-7 h-7 bg-[#F4F4F0] border-[1.5px] border-black rounded-lg flex items-center justify-center font-black"
                >
                  ✕
                </button>
              </div>

              <div className="overflow-y-auto space-y-3 flex-1 pr-1">
                {/* Stats Summary */}
                <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-1">
                  <div className="flex justify-between font-black">
                    <span>CAMERA DRIVER:</span>
                    <span className="uppercase text-[#38BDF8]">{cameraType}</span>
                  </div>
                  <div className="flex justify-between font-black">
                    <span>ENROLLED USERS:</span>
                    <span>{totalUsers}</span>
                  </div>
                  <div className="flex justify-between font-black">
                    <span>THRESHOLD:</span>
                    <span>MNHD &le; 0.3800</span>
                  </div>
                </div>

                {/* Biometric Separation Matrix */}
                <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-display font-black text-xs uppercase">ACCURACY MATRIX</span>
                    <button 
                      onClick={fetchReport} 
                      disabled={reportLoading}
                      className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded text-[10px] font-black"
                    >
                      {reportLoading ? 'Loading...' : 'Refresh'}
                    </button>
                  </div>

                  {reportData?.self_matches && reportData.self_matches.length > 0 ? (
                    <div className="space-y-1">
                      {reportData.self_matches.map(m => (
                        <div key={m.username} className="p-1.5 bg-[#F8F8F4] rounded flex justify-between font-mono text-[10px]">
                          <span>{m.username}</span>
                          <span>avg:{m.avg_score.toFixed(3)} [{m.quality}]</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[#888] italic text-[10px]">Need &ge; 2 enrolled samples to compute matrix.</p>
                  )}
                </div>

                {/* Wipe Database */}
                <button
                  onClick={handleResetDatabase}
                  className="w-full py-2.5 bg-[#FF4081] text-white border-[2.5px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn"
                >
                  🗑️ RESET DATABASE (Wipe All Biometrics)
                </button>
              </div>

              <button
                onClick={() => setAdminOpen(false)}
                className="w-full py-2.5 bg-[#FFDE59] border-[2px] border-black rounded-xl font-display font-black text-xs neo-btn"
              >
                CLOSE ADMIN
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
