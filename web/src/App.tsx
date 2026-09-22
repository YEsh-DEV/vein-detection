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
  Lock,
  Cpu,
  Eye,
  Check,
  Zap,
  Info
} from 'lucide-react';

type AppState = 'idle' | 'scan' | 'enroll';
type CameraState = 'connecting' | 'live' | 'error';

interface ScanResult {
  accepted: boolean;
  username: string | null;
  score: number;
  threshold: number;
  time_ms: number;
  clahe_base64?: string;
}

interface ReportData {
  users?: Array<{ id: number; username: string; sample_count: number; created_at: string }>;
  total_templates?: number;
}

// ── Custom Palm Vein SVG Icon (Sub-dermal vascular tracks & sensor nodes) ──
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

// ── Palm Silhouette Alignment Guide Overlay ──
function PalmSilhouetteGuide({ active = false, warning = '' }: { active?: boolean; warning?: string }) {
  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center p-6 z-10">
      <div className={`relative w-48 h-48 rounded-3xl border-[3px] border-dashed transition-all duration-300 flex items-center justify-center ${
        active ? 'border-[#CCFF00] bg-[#CCFF00]/10 scale-105' : 'border-[#FFDE59]/80 bg-black/30'
      }`}>
        {/* Palm shape ghost stencil */}
        <PalmIcon className={`w-32 h-32 opacity-70 transition-colors ${active ? 'text-[#CCFF00]' : 'text-[#FFDE59]'}`} />
        
        {/* Corner guide brackets */}
        <div className="absolute top-2 left-2 w-5 h-5 border-t-4 border-l-4 border-white" />
        <div className="absolute top-2 right-2 w-5 h-5 border-t-4 border-r-4 border-white" />
        <div className="absolute bottom-2 left-2 w-5 h-5 border-b-4 border-l-4 border-white" />
        <div className="absolute bottom-2 right-2 w-5 h-5 border-b-4 border-r-4 border-white" />

        {/* Distance height guide badge */}
        <div className="absolute -top-3 px-2 py-0.5 bg-[#FFDE59] border border-black rounded text-[9px] font-mono font-black text-black uppercase tracking-wider">
          ↕ 10 - 15 CM DISTANCE
        </div>
      </div>
      <div className="mt-3 px-3 py-1 bg-black/90 border border-white/40 rounded-full text-[10px] font-mono font-bold text-white uppercase tracking-wider backdrop-blur-md shadow-[0_2px_4px_rgba(0,0,0,0.5)]">
        {warning || 'HOLD PALM FLAT • SPREAD FINGERS SLIGHTLY'}
      </div>
    </div>
  );
}

// ── Reusable Camera Viewport ──
function CameraViewport({
  cameraState,
  cameraErrorDetail,
  onRetry,
  children,
  className = "w-full h-full min-h-[300px]",
  showLiveTag = true,
}: {
  cameraState: CameraState;
  cameraErrorDetail?: string;
  onRetry?: () => void;
  children?: React.ReactNode;
  className?: string;
  showLiveTag?: boolean;
}) {
  return (
    <div className={`relative rounded-2xl border-[3.5px] border-black shadow-[4px_4px_0px_#121212] overflow-hidden bg-black flex items-center justify-center ${className}`}>
      {/* STATE 1: CONNECTING */}
      {cameraState === 'connecting' && (
        <div className="flex flex-col items-center justify-center text-center p-5 text-white animate-pulse space-y-2.5">
          <div className="w-12 h-12 rounded-2xl bg-[#1e1e1e] border-[2px] border-[#FFDE59] flex items-center justify-center">
            <RefreshCw className="w-6 h-6 animate-spin text-[#FFDE59]" />
          </div>
          <span className="font-display font-black text-xs uppercase tracking-wider text-[#FFDE59]">
            CONNECTING TO CAMERA...
          </span>
          <span className="text-[10px] text-[#aaa] font-bold">
            Initializing hardware capture pipeline
          </span>
        </div>
      )}

      {/* STATE 2: LIVE FEED */}
      {cameraState === 'live' && (
        <>
          <img 
            src="/api/video_feed" 
            alt="Live Camera Feed" 
            className="w-full h-full object-cover"
          />
          {showLiveTag && (
            <div className="absolute top-2.5 right-2.5 px-2 py-0.5 bg-[#121212]/85 border-[1.5px] border-[#CCFF00] rounded-md text-[9px] font-mono font-black text-[#CCFF00] flex items-center gap-1.5 shadow-sm z-20">
              <span className="w-1.5 h-1.5 rounded-full bg-[#CCFF00] animate-pulse" />
              <span>LIVE SENSOR</span>
            </div>
          )}
        </>
      )}

      {/* STATE 3: ERROR / DETACHED */}
      {cameraState === 'error' && (
        <div className="flex flex-col items-center justify-center text-center p-5 bg-[#250404] text-white w-full h-full space-y-2">
          <div className="w-11 h-11 rounded-2xl bg-[#FF4081] border-[2px] border-black flex items-center justify-center shadow-[2px_2px_0px_#121212]">
            <AlertTriangle className="w-6 h-6 text-white" />
          </div>
          <span className="font-display font-black text-xs uppercase tracking-wider text-[#FF4081]">
            CAMERA HARDWARE NOT DETECTED
          </span>
          <p className="text-[10px] font-bold text-[#FFB0B0] max-w-[240px] leading-tight">
            {cameraErrorDetail || "Camera not detected — check camera cable on Raspberry Pi."}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-1 px-3 py-1.5 bg-[#FFDE59] text-black border-[2px] border-black rounded-xl text-[11px] font-black shadow-[2px_2px_0px_#121212] neo-btn flex items-center gap-1"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Connection</span>
            </button>
          )}
        </div>
      )}

      {children}
    </div>
  );
}

const SAMPLE_GUIDANCE = [
  "Sample 1: Hold palm flat & centered ~10-15cm above sensor",
  "Sample 2: Tilt palm slightly to the LEFT (~5 degrees)",
  "Sample 3: Tilt palm slightly to the RIGHT (~5 degrees)",
  "Sample 4: Raise palm slightly HIGHER (~15-18cm)",
  "Sample 5: Lower palm slightly closer (~10cm)",
  "Sample 6: Hold palm flat for final template confirmation",
];

// ── Error Boundary ──
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center bg-[#250404] text-white">
          <AlertTriangle className="w-12 h-12 text-[#FF4081] mb-3" />
          <h2 className="font-display font-black text-lg">System Notice</h2>
          <p className="text-xs text-[#FFB0B0] mt-2 max-w-[280px]">The kiosk interface encountered an unexpected state. Please tap to restart.</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-[#FFDE59] text-black border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212]"
          >
            Restart Kiosk
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  // 3-Screen Public State Machine
  const [appState, setAppState] = useState<AppState>('idle');

  // Hardware Status
  const [cameraState, setCameraState] = useState<CameraState>('connecting');
  const [cameraErrorDetail, setCameraErrorDetail] = useState<string>('');
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraType, setCameraType] = useState('Checking...');
  const [totalUsers, setTotalUsers] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);

  // Scanning State
  const [isScanning, setIsScanning] = useState(false);
  const [scanCountdown, setScanCountdown] = useState<number | null>(null);
  const [lastScan, setLastScan] = useState<ScanResult | null>(null);
  const [resultOverlay, setResultOverlay] = useState<ScanResult | null>(null);

  // Enrollment State (3-6 Samples)
  const [enrollUsername, setEnrollUsername] = useState('');
  const [enrollSamples, setEnrollSamples] = useState<Array<{ vr_mean: number; thumb: string }>>([]);
  const [isCapturingSample, setIsCapturingSample] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState<number | null>(null);
  const [enrollStatusMsg, setEnrollStatusMsg] = useState('');

  // Hidden Admin / Ops Modal (5-tap trigger on status bead)
  const [adminOpen, setAdminOpen] = useState(false);
  const [adminTapCount, setAdminTapCount] = useState(0);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

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
        const available = Boolean(data.camera_available);
        setCameraReady(available);
        setCameraType(data.camera_type || 'None');
        setTotalUsers(data.enrolled_users_count || 0);
        setModelLoaded(data.model_loaded || false);

        if (available && data.camera_type !== 'None') {
          setCameraState('live');
          setCameraErrorDetail('');
        } else {
          setCameraState('error');
          setCameraErrorDetail('Camera offline — verify cable connection.');
        }
      } else {
        setCameraReady(false);
        setModelLoaded(false);
        setCameraState('error');
        setCameraErrorDetail('Server unavailable.');
      }
    } catch {
      setCameraReady(false);
      setModelLoaded(false);
      setCameraType('Disconnected');
      setCameraState('error');
      setCameraErrorDetail('Cannot reach backend server.');
    }
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  // Timers
  const resultAutoReturnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearResultAutoReturn = () => {
    if (resultAutoReturnTimerRef.current) {
      clearTimeout(resultAutoReturnTimerRef.current);
      resultAutoReturnTimerRef.current = null;
    }
  };

  const closeResultOverlay = (target: 'scan' | 'idle' = 'scan') => {
    clearResultAutoReturn();
    setResultOverlay(null);
    if (target === 'idle') {
      setAppState('idle');
    }
  };

  // Ensure timers are cleared whenever appState changes
  useEffect(() => {
    return () => {
      clearResultAutoReturn();
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
    };
  }, [appState]);

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

  // Idle Timeout: if Scan screen has no interaction for 35s, return to Idle
  useEffect(() => {
    if (appState !== 'scan') return;
    if (isScanning || scanCountdown !== null || resultOverlay !== null) return;

    const startIdleCountdown = () => {
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
      idleTimeoutRef.current = setTimeout(() => {
        if (appStateRef.current === 'scan') {
          setAppState('idle');
        }
      }, 35000);
    };

    startIdleCountdown();
    const resetOnUserActivity = () => startIdleCountdown();

    window.addEventListener('touchstart', resetOnUserActivity, { passive: true });
    window.addEventListener('mousedown', resetOnUserActivity, { passive: true });

    return () => {
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
      window.removeEventListener('touchstart', resetOnUserActivity);
      window.removeEventListener('mousedown', resetOnUserActivity);
    };
  }, [appState, isScanning, scanCountdown, resultOverlay]);

  // Scan Execution with 3-Second Countdown
  const handleScanWithCountdown = async () => {
    if (isScanning || scanCountdown !== null || !cameraReady) return;
    clearResultAutoReturn();
    
    // 3-Second countdown
    for (let i = 3; i > 0; i--) {
      if (appStateRef.current !== 'scan') {
        setScanCountdown(null);
        return;
      }
      setScanCountdown(i);
      await new Promise(r => setTimeout(r, 900));
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
            particleCount: 100,
            spread: 75,
            origin: { y: 0.55 },
            colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00', '#121212']
          });
        }

        // Auto-return to Idle state after 3.8s
        clearResultAutoReturn();
        resultAutoReturnTimerRef.current = setTimeout(() => {
          closeResultOverlay('idle');
        }, 3800);

      } else {
        const err = await res.json().catch(() => ({}));
        const rawDetail = err.detail || 'Palm not recognized';
        let friendlyMsg = rawDetail;
        if (rawDetail.includes('farther') || rawDetail.includes('too close')) {
          friendlyMsg = '⚠️ Hand too close — move hand farther (~10-15cm) from lens.';
        } else if (rawDetail.includes('closer') || rawDetail.includes('too far')) {
          friendlyMsg = '⚠️ Hand too far — move hand closer to sensor.';
        } else if (rawDetail.includes('center') || rawDetail.includes('outside')) {
          friendlyMsg = '⚠️ Center palm within the guide frame.';
        } else if (rawDetail.includes('valleys') || rawDetail.includes('fingers')) {
          friendlyMsg = '⚠️ Spread fingers slightly to expose knuckle valleys.';
        } else if (rawDetail.includes('Quality gate')) {
          friendlyMsg = `⚠️ ${rawDetail}`;
        }
        showToast(friendlyMsg, 'warn');
        setResultOverlay({
          accepted: false,
          username: null,
          score: 0.0,
          threshold: 0.2226,
          time_ms: 50
        });
        clearResultAutoReturn();
        resultAutoReturnTimerRef.current = setTimeout(() => {
          closeResultOverlay('scan');
        }, 3000);
      }
    } catch {
      showToast('Terminal connection error. Please retry.', 'error');
    } finally {
      setIsScanning(false);
    }
  };

  // Live Camera Sample Capture with 5-Second Countdown
  const handleCaptureSampleWithCountdown = async () => {
    if (isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !cameraReady) return;
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (!cleanUname) {
      showToast('Enter a username or identifier first!', 'warn');
      return;
    }

    const currentHint = SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm steady ~10-15cm above sensor';
    setEnrollStatusMsg(`${currentHint} (Capturing in 3 seconds...)`);

    // 3-Second Countdown for responsive enrollment
    for (let i = 3; i > 0; i--) {
      if (appStateRef.current !== 'enroll') {
        setEnrollCountdown(null);
        return;
      }
      setEnrollCountdown(i);
      await new Promise(r => setTimeout(r, 900));
    }
    if (appStateRef.current !== 'enroll') {
      setEnrollCountdown(null);
      return;
    }
    setEnrollCountdown(null);
    setIsCapturingSample(true);
    setEnrollStatusMsg('Acquiring frame & validating palm alignment...');

    try {
      const res = await fetch('/api/enroll/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUname, sample_idx: enrollSamples.length }),
      });
      if (res.ok) {
        const data = await res.json();
        setEnrollSamples(prev => [...prev, { vr_mean: data.vr_mean || 0.5, thumb: data.thumb || '' }]);
        const nextIdx = enrollSamples.length + 1;
        const nextHint = SAMPLE_GUIDANCE[nextIdx] || 'Ready to finalize template.';
        setEnrollStatusMsg(`Sample #${nextIdx} captured! ${nextIdx >= 3 ? 'You can finish or capture more.' : nextHint}`);
        showToast(`Sample ${nextIdx}/6 captured!`, 'success');
        const err = await res.json().catch(() => ({}));
        const rawDetail = err.detail || 'Hand not detected';
        let friendlyMsg = rawDetail;
        if (rawDetail.includes('farther') || rawDetail.includes('too close')) {
          friendlyMsg = 'Hand too close — move hand farther (~10-15cm) from lens.';
        } else if (rawDetail.includes('closer') || rawDetail.includes('too far')) {
          friendlyMsg = 'Hand too far — move hand closer to sensor.';
        } else if (rawDetail.includes('center') || rawDetail.includes('outside')) {
          friendlyMsg = 'Center palm within the guide frame.';
        } else if (rawDetail.includes('valleys') || rawDetail.includes('fingers')) {
          friendlyMsg = 'Spread fingers slightly and keep hand flat.';
        } else if (rawDetail.includes('Quality gate')) {
          friendlyMsg = rawDetail;
        }
        setEnrollStatusMsg(`Sample rejected: ${friendlyMsg}`);
        showToast(friendlyMsg, 'warn');
      }
    } catch {
      showToast('Terminal connection error.', 'error');
      setEnrollStatusMsg('Error communicating with terminal backend.');
    } finally {
      setIsCapturingSample(false);
    }
  };

  // Commit Multi-Sample Enrollment to Database
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
        showToast(`Enrolled '${cleanUname}' successfully!`, 'success');
        confetti({
          particleCount: 120,
          spread: 85,
          origin: { y: 0.5 },
          colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00']
        });
        setEnrollUsername('');
        setEnrollSamples([]);
        setEnrollStatusMsg('');
        loadStatus();
        setTimeout(() => setAppState('idle'), 1500);
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Failed to save enrollment to database.', 'error');
      }
    } catch {
      showToast('Terminal connection error saving template.', 'error');
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

  const handleDeleteUser = async (uname: string) => {
    if (!window.confirm(`Delete user '${uname}' and remove all templates?`)) return;
    try {
      const res = await fetch(`/api/users/${uname}`, { method: 'DELETE' });
      if (res.ok) {
        showToast(`User '${uname}' removed.`, 'success');
        fetchReport();
        loadStatus();
      }
    } catch {
      showToast('Error removing user.', 'error');
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-dribbble-yellow flex justify-center items-center p-0 sm:p-4 text-[#121212] select-none font-sans">
       
        {/* ── 5" RASPBERRY PI TOUCH DISPLAY FRAME (800×480 LANDSCAPE OR 720×1280 PORTRAIT) ── */}
        <div className="w-full max-w-[760px] h-[98vh] max-h-[1200px] bg-[#FFFDF0] border-[4px] border-black rounded-[24px] shadow-[8px_8px_0px_#121212] flex flex-col relative overflow-hidden bg-neo-cream">

          {/* ── SERVICE STATUS HEADER (Minimal, Non-Intrusive) ── */}
          <div className="px-5 pt-3 pb-2.5 flex items-center justify-between text-xs font-black text-black z-20 border-b-[2px] border-black/10">
            <div className="flex items-center gap-2">
              {/* Secret 5-tap Admin trigger on camera bead */}
              <button 
                onClick={handleSecretAdminTap}
                className="flex items-center gap-1.5 focus:outline-none cursor-pointer"
                title="Status indicator"
              >
                <span className={`w-3 h-3 rounded-full border-[1.5px] border-black transition-colors ${
                  cameraState === 'live' ? 'bg-[#CCFF00]' : cameraState === 'connecting' ? 'bg-[#FFDE59] animate-ping' : 'bg-[#FF4081]'
                }`} />
                <span className="text-[10px] uppercase font-bold tracking-tight text-[#444]">{cameraType}</span>
              </button>
            </div>

            <div className="text-center font-display font-black text-xs tracking-wider uppercase">
              PALM BIOMETRIC KIOSK
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
              - Offline Neo-Brutalist hero/landing experience
              - ANY touch/click skips immediately into the Scan screen
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'idle' && (
            <div 
              onClick={() => setAppState('scan')}
              className="flex-1 flex flex-col items-center justify-between p-6 cursor-pointer animate-fadeIn relative overflow-hidden select-none"
            >
              {/* Top Status */}
              <div className="w-full flex justify-between items-center z-10">
                <div className="px-3 py-1 bg-white border-[2px] border-black rounded-xl text-[10px] font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#CCFF00] animate-pulse" />
                  <span>TERMINAL READY</span>
                </div>

                <div className="px-3 py-1 bg-[#FFDE59] border-[2px] border-black rounded-xl text-[10px] font-black shadow-[2px_2px_0px_#121212]">
                  {totalUsers} ENROLLED
                </div>
              </div>

              {/* Main Center "Ready" Card */}
              <div className="w-full bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-6 shadow-[8px_8px_0px_#121212] flex flex-col items-center text-center my-auto z-10 neo-btn">
               
                {/* Animated Vein Sensor Graphic */}
                <div className="relative my-3">
                  <div className="w-28 h-28 rounded-3xl bg-[#CCFF00] border-[3.5px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center animate-float">
                    <PalmIcon className="w-16 h-16 text-black" animated={true} />
                  </div>
                  <div className="absolute -bottom-2 -right-2 px-2.5 py-1 bg-[#FF4081] text-white border-[2px] border-black rounded-lg text-[9px] font-black shadow-[2px_2px_0px_#121212]">
                    ⚡ CONTACTLESS
                  </div>
                </div>

                <div className="space-y-2 mt-2">
                  <h1 className="font-display font-black text-2xl leading-tight uppercase tracking-tight">
                    TOUCH SCREEN TO SCAN PALM
                  </h1>
                  <p className="text-xs font-bold text-[#666] max-w-[280px] mx-auto">
                    Hold palm 10-15cm flat above the camera sensor to authenticate.
                  </p>
                </div>

                {/* High-visibility Action Prompt */}
                <div className="mt-5 w-full py-4 bg-[#FFDE59] border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-base flex items-center justify-center gap-2 animate-pulse">
                  <span>TAP ANYWHERE TO START</span>
                  <ArrowRight className="w-5 h-5 stroke-[3]" />
                </div>
              </div>

              {/* Bottom Instruction */}
              <div className="text-center z-10">
                <span className="text-[10px] font-black text-[#555] uppercase tracking-wider">
                  Touch screen to begin • Auto-timeout 35s • Prototype Biometric Terminal
                </span>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              SCREEN 2: SCAN (DOMINANT CAMERA VIEWPORT WITH GUIDANCE)
              - Dominant camera feed
              - Clear alignment silhouette guide
              - Visual cues: Move closer, align palm, hold steady
              - Optimized for 5-inch display without scrolling
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'scan' && (
            <div className="flex-1 flex flex-col p-4 space-y-3 animate-fadeIn overflow-hidden">
             
              {/* Header Bar: Cancel on left, Mode in center, Enroll on right */}
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setAppState('idle')}
                  className="px-3 py-1.5 bg-white border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1 neo-btn"
                >
                  <ArrowLeft className="w-3.5 h-3.5 stroke-[3]" />
                  <span>Exit</span>
                </button>

                <div className="px-3 py-1 bg-[#FFDE59] border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>PALM SCAN MODE</span>
                </div>

                {/* Subtle trigger to access Enrollment for authorized users */}
                <button
                  onClick={() => setAppState('enroll')}
                  className="px-2.5 py-1 bg-[#F4F4F0] border-[2px] border-black rounded-xl shadow-[2px_2px_0px_#121212] flex items-center gap-1 text-xs font-black hover:text-black neo-btn"
                  title="Enroll New Palm"
                >
                  <UserPlus className="w-3.5 h-3.5" />
                  <span>Enroll</span>
                </button>
              </div>

              {/* Dominant Camera Viewport */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-2.5 shadow-[5px_5px_0px_#121212] relative overflow-hidden flex flex-col items-center flex-1 justify-center">
                <CameraViewport 
                  cameraState={cameraState}
                  cameraErrorDetail={cameraErrorDetail}
                  onRetry={loadStatus}
                  className="w-full h-full min-h-[280px]"
                >
                  {/* Palm Silhouette Guidance Overlay */}
                  <PalmSilhouetteGuide active={scanCountdown !== null || isScanning} />

                  {/* 3-Second Countdown Overlay */}
                  {scanCountdown !== null && (
                    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center animate-fadeIn z-30">
                      <span className="font-display font-black text-8xl text-[#FFDE59] drop-shadow-[4px_4px_0px_#000] animate-bounce">
                        {scanCountdown}
                      </span>
                      <span className="text-xs font-black text-white bg-black/90 px-3 py-1 rounded-md mt-2 tracking-wider border border-white/20">
                        HOLD PALM STEADY
                      </span>
                    </div>
                  )}

                  {/* Processing Overlay */}
                  {isScanning && (
                    <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center animate-fadeIn text-white z-30 space-y-2">
                      <RefreshCw className="w-10 h-10 animate-spin text-[#38BDF8]" />
                      <span className="text-sm font-black text-[#CCFF00] tracking-wider uppercase">EXTRACTING VEIN PATTERN...</span>
                    </div>
                  )}
                </CameraViewport>
              </div>

              {/* Status Bead Guidance Indicator */}
              <div className="text-center">
                <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-black border-[2px] border-black shadow-[2px_2px_0px_#121212] ${
                  scanCountdown !== null 
                    ? 'bg-[#FFDE59] text-black animate-pulse' 
                    : isScanning 
                    ? 'bg-[#38BDF8] text-black' 
                    : cameraState === 'live' 
                    ? 'bg-[#CCFF00] text-black' 
                    : 'bg-[#FF4081] text-white'
                }`}>
                  {scanCountdown !== null 
                    ? `HOLD STEADY (${scanCountdown}s)...` 
                    : isScanning 
                    ? 'VERIFYING BIOMETRICS...' 
                    : cameraState === 'live' 
                    ? 'READY — ALIGN PALM AND TAP SCAN' 
                    : 'CAMERA OFFLINE'}
                </span>
              </div>

              {/* Primary Action Button (Large Touch Target) */}
              <button
                onClick={handleScanWithCountdown}
                disabled={isScanning || scanCountdown !== null || cameraState !== 'live' || !modelLoaded}
                className="w-full py-4 bg-[#FFDE59] text-black border-[3.5px] border-black rounded-2xl shadow-[5px_5px_0px_#121212] font-display font-black text-lg flex items-center justify-center gap-3 neo-btn hover:bg-[#ffe26b] disabled:bg-[#E2E8F0] disabled:text-[#888888] disabled:border-[#888888] disabled:shadow-none disabled:cursor-not-allowed"
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
                ) : cameraState !== 'live' ? (
                  <>
                    <AlertTriangle className="w-5 h-5 text-[#888]" />
                    <span>CAMERA OFFLINE</span>
                  </>
                ) : !modelLoaded ? (
                  <>
                    <AlertTriangle className="w-5 h-5 text-[#888]" />
                    <span>MODEL NOT LOADED</span>
                  </>
                ) : (
                  <>
                    <Eye className="w-6 h-6 stroke-[3]" />
                    <span>SCAN PALM NOW</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              SCREEN 3: ENROLLMENT (3–6 SAMPLES GUIDED STUDIO)
              - Clear sample counter: Sample 1/6, 2/6...
              - Allows finishing at 3 samples or continuing up to 6
              - Shows preview thumbnails of ROIs
              - Zero technical jargon
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'enroll' && (
            <div className="flex-1 flex flex-col p-4 space-y-3 animate-fadeIn overflow-y-auto pb-4">
             
              {/* Header: Title + Exit Button */}
              <div className="flex items-center justify-between border-b-[2px] border-black/10 pb-2">
                <div>
                  <h2 className="font-display font-black text-base tracking-tight uppercase">ENROLL PALM</h2>
                  <p className="text-[10px] font-bold text-[#666]">Guided 3–6 sample biometric calibration</p>
                </div>

                <button
                  onClick={() => setAppState('idle')}
                  className="w-8 h-8 rounded-xl bg-[#F4F4F0] border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-black text-sm neo-btn"
                  title="Exit"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Identifier Input Field */}
              <div className="space-y-1">
                <label className="text-[10px] font-black uppercase tracking-wider text-black">USERNAME / IDENTIFIER</label>
                <input
                  type="text"
                  value={enrollUsername}
                  onChange={e => setEnrollUsername(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                  placeholder="e.g. user_alpha"
                  className="w-full px-3.5 py-2 bg-white border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm outline-none focus:bg-[#FFFDF0]"
                />
              </div>

              {/* 6-Cell Sample Matrix Grid */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-2.5 shadow-[3px_3px_0px_#121212] space-y-1.5">
                <div className="flex justify-between items-center text-xs font-black">
                  <span className="uppercase tracking-wider">CALIBRATION SAMPLES:</span>
                  <span className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded-full text-[10px]">
                    {enrollSamples.length} / 6 SAMPLES {enrollSamples.length >= 3 ? '(READY)' : ''}
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
                          isDone ? 'bg-[#CCFF00] scale-105' : idx < 3 ? 'bg-[#FFFDE8] text-[#888]' : 'bg-[#F4F4F0] text-[#aaa]'
                        }`}>
                        {isDone && sample.thumb ? (
                          <img src={`data:image/png;base64,${sample.thumb}`} alt={`Sample ${idx+1}`} className="w-full h-full object-cover" />
                        ) : isDone ? (
                          '✓'
                        ) : (
                          `#${idx + 1}`
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Mini Camera Viewport with Countdown */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-2 shadow-[4px_4px_0px_#121212] relative overflow-hidden flex flex-col items-center">
                <CameraViewport
                  cameraState={cameraState}
                  cameraErrorDetail={cameraErrorDetail}
                  onRetry={loadStatus}
                  className="w-48 h-32"
                  showLiveTag={false}
                >
                  {enrollCountdown !== null && (
                    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center z-10">
                      <span className="font-display font-black text-5xl text-[#FFDE59] drop-shadow-[2px_2px_0px_#000] animate-bounce">
                        {enrollCountdown}
                      </span>
                    </div>
                  )}

                  {isCapturingSample && (
                    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white z-10">
                      <RefreshCw className="w-6 h-6 animate-spin text-[#38BDF8]" />
                    </div>
                  )}
                </CameraViewport>
              </div>

              {/* Sample Capture Action Button */}
              <button
                onClick={handleCaptureSampleWithCountdown}
                disabled={isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !enrollUsername.trim() || cameraState !== 'live' || !modelLoaded}
                className={`w-full py-3 border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm flex items-center justify-center gap-2 neo-btn disabled:bg-[#E2E8F0] disabled:text-[#888888] disabled:border-[#888888] disabled:shadow-none disabled:cursor-not-allowed ${
                  enrollCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : 'bg-[#38BDF8] text-black hover:bg-[#2cb0eb]'
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
                    <span>PROCESSING SAMPLE...</span>
                  </>
                ) : (
                  <>
                    <Camera className="w-5 h-5 stroke-[2.5]" />
                    <span>CAPTURE SAMPLE #{enrollSamples.length + 1}</span>
                  </>
                )}
              </button>

              {/* Guidance Banner */}
              <div className="border-[2px] border-black rounded-xl p-2.5 shadow-[2px_2px_0px_#121212] bg-[#FFDE59] text-black">
                <p className="text-[11px] font-bold leading-tight flex items-center gap-1.5">
                  <Info className="w-4 h-4 shrink-0" />
                  <span>{enrollStatusMsg || SAMPLE_GUIDANCE[enrollSamples.length] || 'Hold palm flat ~10-15cm above sensor.'}</span>
                </p>
              </div>

              {/* Save & Commit Button */}
              {enrollSamples.length >= 3 && (
                <button
                  onClick={handleSaveEnrollment}
                  className="w-full py-3.5 bg-[#CCFF00] text-black border-[3.5px] border-black rounded-xl shadow-[4px_4px_0px_#121212] font-display font-black text-sm flex items-center justify-center gap-2 neo-btn hover:bg-[#b8e600] animate-bounce"
                >
                  <Check className="w-5 h-5 stroke-[3]" />
                  <span>SAVE & FINISH ENROLLMENT ({enrollSamples.length} SAMPLES)</span>
                </button>
              )}
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              FULLSCREEN RESULT OVERLAY (SUCCESS / FAILURE MOMENT)
              - Vibrant Neo-Brutalist green for match / coral for rejection
              - Personalized confirmation
              - Auto-returns after 3.8s
             ══════════════════════════════════════════════════════════════════════ */}
          {resultOverlay && (
            <div className={`absolute inset-0 z-50 p-6 flex flex-col items-center justify-center animate-fadeIn ${
              resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081]'
            }`}>
              <div className="w-full max-w-[460px] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-6 shadow-[8px_8px_0px_#121212] text-center space-y-4 neo-card">
               
                {/* Giant Illuminated Status Icon */}
                <div className="relative mx-auto w-20 h-20">
                  <div className={`w-20 h-20 rounded-2xl border-[3.5px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center font-display font-black text-4xl animate-float ${
                    resultOverlay.accepted ? 'bg-[#CCFF00] text-black' : 'bg-[#FF4081] text-white'
                  }`}>
                    {resultOverlay.accepted ? '✓' : '✕'}
                  </div>
                </div>

                {/* Headline & Identity */}
                <div className="space-y-1">
                  <h3 className="font-display font-black text-2xl tracking-tight uppercase leading-tight">
                    {resultOverlay.accepted ? 'PALM VERIFIED' : 'NOT RECOGNIZED'}
                  </h3>
                  <p className="font-bold text-xs text-[#555]">
                    {resultOverlay.accepted 
                      ? `Welcome, ${resultOverlay.username?.toUpperCase()}! Identity confirmed.` 
                      : 'Vascular vein pattern could not be recognized. Please reposition hand and try again.'}
                  </p>
                </div>

                {/* Telemetry Card */}
                <div className="bg-white border-[2px] border-black rounded-2xl p-3 shadow-[2px_2px_0px_#121212] space-y-1.5 text-left">
                  <div className="flex justify-between items-center text-xs font-black">
                    <span className="text-[#666] uppercase">SCAN RESULT:</span>
                    <span className={`px-2 py-0.5 rounded border-[1.5px] border-black text-[10px] ${
                      resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                    }`}>
                      {resultOverlay.accepted ? 'MATCH CONFIRMED' : 'REJECTED'}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-[10px] font-mono font-bold text-[#666]">
                    <span>Latency: {resultOverlay.time_ms} ms</span>
                    <span>Status: Prototype Terminal</span>
                  </div>

                  {resultOverlay.clahe_base64 && (
                    <div className="pt-2 border-t-[1.5px] border-black/10 flex items-center justify-between">
                      <span className="text-[10px] font-black text-[#555] uppercase">EXTRACTED VEIN ROI:</span>
                      <img 
                        src={`data:image/png;base64,${resultOverlay.clahe_base64}`} 
                        alt="Enhanced ROI" 
                        className="w-8 h-8 rounded-lg border-[1.5px] border-black object-cover bg-black"
                      />
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <button
                    onClick={() => closeResultOverlay('scan')}
                    className="py-3 bg-[#CCFF00] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#b8e600] flex items-center justify-center gap-1.5"
                  >
                    <RefreshCw className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>SCAN AGAIN</span>
                  </button>

                  <button
                    onClick={() => closeResultOverlay('idle')}
                    className="py-3 bg-[#FFDE59] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#ffe26b] flex items-center justify-center gap-1.5"
                  >
                    <ArrowLeft className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>EXIT (3s)</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              HIDDEN OPERATOR / ADMIN MODAL
              - Accessible only via 5 rapid taps on top-left camera indicator bead
              - Zero exposure in public flow
             ══════════════════════════════════════════════════════════════════════ */}
          {adminOpen && (
            <div className="absolute inset-0 bg-black/80 z-50 flex items-center justify-center p-4 animate-fadeIn">
              <div className="w-full max-h-[92%] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-5 shadow-[8px_8px_0px_#121212] flex flex-col space-y-3 overflow-hidden text-xs">
               
                <div className="flex justify-between items-center border-b-2 border-black pb-2">
                  <div className="flex items-center gap-2">
                    <Lock className="w-4 h-4 text-black" />
                    <h3 className="font-display font-black text-sm uppercase">TERMINAL OPERATOR SETTINGS</h3>
                  </div>
                  <button 
                    onClick={() => setAdminOpen(false)} 
                    className="w-7 h-7 bg-[#F4F4F0] border-[1.5px] border-black rounded-lg flex items-center justify-center font-black"
                  >
                    ✕
                  </button>
                </div>

                <div className="overflow-y-auto space-y-3 flex-1 pr-1">
                  {/* System & Hardware Specs */}
                  <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-1.5 shadow-[2px_2px_0px_#121212]">
                    <div className="flex items-center gap-1.5 text-[#38BDF8] font-black pb-1 border-b border-black/10">
                      <Cpu className="w-3.5 h-3.5" />
                      <span className="uppercase text-[11px]">HARDWARE & BIOMETRIC ENGINE</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>ENGINE:</span>
                      <span className="font-mono">AMPVNet ONNX (v2)</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>CAMERA:</span>
                      <span className="uppercase text-[#38BDF8]">{cameraType}</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>ENROLLED IDENTITIES:</span>
                      <span className="font-mono">{totalUsers} Users</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>EXPERIMENTAL THRESHOLD:</span>
                      <span className="font-mono">0.2226</span>
                    </div>
                  </div>

                  {/* Registered Users Management */}
                  <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-2 shadow-[2px_2px_0px_#121212]">
                    <div className="flex justify-between items-center">
                      <span className="font-display font-black text-xs uppercase">REGISTERED USERS</span>
                      <button 
                        onClick={fetchReport} 
                        disabled={reportLoading}
                        className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded text-[10px] font-black neo-btn"
                      >
                        {reportLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>

                    {reportData?.users && reportData.users.length > 0 ? (
                      <div className="space-y-1 max-h-40 overflow-y-auto">
                        {reportData.users.map(u => (
                          <div key={u.username} className="p-1.5 bg-[#F8F8F4] border border-black/10 rounded flex justify-between items-center font-mono text-[10px]">
                            <span>{u.username} ({u.sample_count} samples)</span>
                            <button
                              onClick={() => handleDeleteUser(u.username)}
                              className="px-1.5 py-0.5 bg-[#FF4081] text-white border border-black rounded text-[9px] font-bold"
                            >
                              Delete
                            </button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-[10px] font-bold text-[#888]">No enrolled users in database.</p>
                    )}
                  </div>
                </div>

                <div className="pt-2 border-t-2 border-black flex justify-end">
                  <button
                    onClick={() => setAdminOpen(false)}
                    className="px-4 py-2 bg-[#FFDE59] border-[2px] border-black rounded-xl font-black text-xs shadow-[2px_2px_0px_#121212] neo-btn"
                  >
                    Close Operator Panel
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </ErrorBoundary>
  );
}