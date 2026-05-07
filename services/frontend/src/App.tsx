import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { DashboardPage } from '@/pages/Dashboard';
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
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
