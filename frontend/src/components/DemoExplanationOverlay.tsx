// frontend/src/components/DemoExplanationOverlay.tsx - Enhanced Demo Explanation UI
import React, { useState, useEffect } from 'react';
import { AlertTriangle, Info, CheckCircle, Shield, FileText, X } from 'lucide-react';

interface DemoExplanation {
  title: string;
  message: string;
  type: 'pattern' | 'question' | 'action' | 'risk' | 'compliance';
  riskLevel?: number;
  duration?: number;
}

interface DemoExplanationOverlayProps {
  explanation: DemoExplanation | null;
  onDismiss?: () => void;
  showRiskBreakdown?: boolean;
}

export const DemoExplanationOverlay: React.FC<DemoExplanationOverlayProps> = ({
  explanation,
  onDismiss,
  showRiskBreakdown = true
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [autoHideTimer, setAutoHideTimer] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (explanation) {
      setIsVisible(true);
      
      // Auto-hide after duration
      if (explanation.duration && explanation.duration > 0) {
        const timer = setTimeout(() => {
          handleDismiss();
        }, explanation.duration);
        setAutoHideTimer(timer);
      }
    } else {
      setIsVisible(false);
    }
    
    return () => {
      if (autoHideTimer) {
        clearTimeout(autoHideTimer);
      }
    };
  }, [explanation]);

  const handleDismiss = () => {
    setIsVisible(false);
    if (autoHideTimer) {
      clearTimeout(autoHideTimer);
      setAutoHideTimer(null);
    }
    onDismiss?.();
  };

  if (!explanation || !isVisible) {
    return null;
  }

  const getTypeConfig = (type: string) => {
    switch (type) {
      case 'pattern':
        return {
          color: 'bg-blue-500',
          icon: <Info className="w-5 h-5" />,
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          textColor: 'text-blue-900'
        };
      case 'question':
        return {
          color: 'bg-orange-500',
          icon: <AlertTriangle className="w-5 h-5" />,
          bgColor: 'bg-orange-50',
          borderColor: 'border-orange-200',
          textColor: 'text-orange-900'
        };
      case 'action':
        return {
          color: 'bg-green-500',
          icon: <CheckCircle className="w-5 h-5" />,
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          textColor: 'text-green-900'
        };
      case 'risk':
        return {
          color: 'bg-red-500',
          icon: <Shield className="w-5 h-5" />,
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          textColor: 'text-red-900'
        };
      case 'compliance':
        return {
          color: 'bg-purple-500',
          icon: <FileText className="w-5 h-5" />,
          bgColor: 'bg-purple-50',
          borderColor: 'border-purple-200',
          textColor: 'text-purple-900'
        };
      default:
        return {
          color: 'bg-gray-500',
          icon: <Info className="w-5 h-5" />,
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          textColor: 'text-gray-900'
        };
    }
  };

  const typeConfig = getTypeConfig(explanation.type);

  return (
    <div className="fixed top-20 left-4 z-50 w-96 animate-in slide-in-from-left duration-300">
      <div className={`${typeConfig.bgColor} ${typeConfig.borderColor} border-2 rounded-lg shadow-lg overflow-hidden`}>
        {/* Header */}
        <div className={`${typeConfig.color} text-white p-3`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {typeConfig.icon}
              <h4 className="font-semibold text-sm">{explanation.title}</h4>
            </div>
            <div className="flex items-center space-x-2">
              {explanation.riskLevel !== undefined && showRiskBreakdown && (
                <span className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">
                  {explanation.riskLevel}% Risk
                </span>
              )}
              <span className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">
                DEMO
              </span>
              <button
                onClick={handleDismiss}
                className="text-white hover:bg-white hover:bg-opacity-20 rounded p-1 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-4">
          <p className={`text-sm leading-relaxed ${typeConfig.textColor}`}>
            {explanation.message}
          </p>
          
          {explanation.riskLevel !== undefined && showRiskBreakdown && (
            <div className="mt-3 pt-3 border-t border-current border-opacity-20">
              <div className="text-xs">
                <div className="font-medium mb-1">Banking Compliance:</div>
                <div className="opacity-90">
                  {explanation.riskLevel >= 80 ? 'CRITICAL - Mandatory intervention required (FCA Guidelines)' :
                   explanation.riskLevel >= 60 ? 'HIGH RISK - Enhanced due diligence required' :
                   explanation.riskLevel >= 40 ? 'MEDIUM RISK - Additional verification recommended' :
                   'LOW RISK - Standard processing permitted'}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Progress Bar (if duration is set) */}
        {explanation.duration && explanation.duration > 0 && (
          <div className="h-1 bg-white bg-opacity-30">
            <div 
              className="h-full bg-white bg-opacity-60 transition-all linear"
              style={{
                animation: `shrink ${explanation.duration}ms linear`,
                width: '100%'
              }}
            />
          </div>
        )}
      </div>
      
      <style jsx>{`
        @keyframes shrink {
          from { width: 100%; }
          to { width: 0%; }
        }
      `}</style>
    </div>
  );
};

export default DemoExplanationOverlay;
