import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { DashboardPage } from '@/pages/Dashboard';
import { LiveWFPage } from '@/pages/LiveWF';
import { LiveIbkrPage } from '@/pages/LiveIbkr';
import { LiveAlpacaPage } from '@/pages/LiveAlpaca';
import { PaperTradePage } from '@/pages/PaperTrade';
import { PerformancePage } from '@/pages/Performance';
import { SettingsPage } from '@/pages/Settings';
import { StockDetailPage } from '@/pages/StockDetail';

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/stocks/:universe/:symbol" element={<StockDetailPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/live-wf" element={<LiveWFPage />} />
        <Route path="/paper" element={<PaperTradePage />} />
        <Route path="/live-ibkr" element={<LiveIbkrPage />} />
        <Route path="/live-alpaca" element={<LiveAlpacaPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
