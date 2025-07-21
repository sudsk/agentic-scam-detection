## File: frontend/src/components/DemoControlPanel.tsx

```typescript
// frontend/src/components/DemoControlPanel.tsx - Enhanced Demo Control Panel
import React from 'react';
import { Play, Pause, Square, Settings, Eye, EyeOff, Zap, ZapOff } from 'lucide-react';

interface DemoControlPanelProps {
  isEnabled: boolean;
  isRunning: boolean;
  currentScript: string | null;
  progress: number;
  autoSimulate: boolean;
  showExplanations: boolean;
  showRiskBreakdown: boolean;
  onToggleDemo: () => void;
  onToggleAutoSimulate: () => void;
  onToggleExplanations: () => void;
  onToggleRiskBreakdown: () => void;
  onStopDemo: () => void;
  nextEvent?: string | null;
  riskScore?: number;
}

export const DemoControlPanel: React.FC<DemoControlPanelProps> = ({
  isEnabled,
  isRunning,
  currentScript,
  progress,
  autoSimulate,
  showExplanations,
  showRiskBreakdown,
  onToggleDemo,
  onToggleAutoSimulate,
  onToggleExplanations,
  onToggleRiskBreakdown,
  onStopDemo,
  nextEvent,
  riskScore = 0
}) => {
  if (!isEnabled) return null;

  return (
    <div className="fixed bottom-4 left-4 z-40 bg-white border-2 border-blue-300 rounded-lg p-4 shadow-lg max-w-sm">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-sm text-gray-900">Demo Control</h4>
        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium">
          DEMO MODE
        </span>
      </div>
      
      {/* Current Status */}
      <div className="space-y-2 text-sm mb-4">
        <div className="flex justify-between">
          <span className="text-gray-600">Script:</span>
          <span className="font-medium text-right flex-1 ml-2 truncate">
            {currentScript || 'None'}
          </span>
        </div>
        
        {isRunning && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-600">Progress:</span>
              <span className="font-medium">{Math.round(progress * 100)}%</span>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            
            {riskScore > 0 && (
              <div className="flex justify-between">
                <span className="text-gray-600">Current Risk:</span>
                <span className={`font-medium px-2 py-1 rounded text-xs ${
                  riskScore >= 80 ? 'bg-red-100 text-red-800' :
                  riskScore >= 60 ? 'bg-orange-100 text-orange-800' :
                  riskScore >= 40 ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {riskScore}%
                </span>
              </div>
            )}
            
            {nextEvent && (
              <div className="bg-blue-50 border border-blue-200 rounded p-2 mt-2">
                <div className="text-xs font-medium text-blue-800 mb-1">Next Event:</div>
                <div className="text-xs text-blue-700">{nextEvent}</div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Control Buttons */}
      <div className="space-y-2">
        {/* Demo Controls */}
        <div className="flex space-x-2">
          {isRunning ? (
            <button
              onClick={onStopDemo}
              className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-red-100 text-red-700 rounded text-xs hover:bg-red-200 transition-colors"
            >
              <Square className="w-3 h-3" />
              <span>Stop Demo</span>
            </button>
          ) : (
            <button
              onClick={onToggleDemo}
              className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-green-100 text-green-700 rounded text-xs hover:bg-green-200 transition-colors"
            >
              <Play className="w-3 h-3" />
              <span>Resume Demo</span>
            </button>
          )}
        </div>

        {/* Feature Toggles */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={onToggleExplanations}
            className={`flex items-center justify-center space-x-1 px-2 py-2 text-xs rounded transition-colors ${
              showExplanations 
                ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {showExplanations ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            <span>Explain</span>
          </button>
          
          <button
            onClick={onToggleAutoSimulate}
            className={`flex items-center justify-center space-x-1 px-2 py-2 text-xs rounded transition-colors ${
              autoSimulate 
                ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {autoSimulate ? <Zap className="w-3 h-3" /> : <ZapOff className="w-3 h-3" />}
            <span>Auto</span>
          </button>
        </div>
        
        <button
          onClick={onToggleRiskBreakdown}
          className={`w-full flex items-center justify-center space-x-1 px-2 py-2 text-xs rounded transition-colors ${
            showRiskBreakdown 
              ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <Settings className="w-3 h-3" />
          <span>Risk Breakdown</span>
        </button>
      </div>

      {/* Demo Info */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="text-xs text-gray-600">
          <div className="font-medium mb-1">Demo Features:</div>
          <ul className="space-y-1">
            <li className="flex items-center space-x-1">
              <span className={showExplanations ? 'text-blue-600' : 'text-gray-400'}>•</span>
              <span>Real-time explanations</span>
            </li>
            <li className="flex items-center space-x-1">
              <span className={autoSimulate ? 'text-green-600' : 'text-gray-400'}>•</span>
              <span>Auto-agent simulation</span>
            </li>
            <li className="flex items-center space-x-1">
              <span className={showRiskBreakdown ? 'text-purple-600' : 'text-gray-400'}>•</span>
              <span>Risk score breakdown</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DemoControlPanel;
```
