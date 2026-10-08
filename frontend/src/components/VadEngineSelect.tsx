import { useConfig } from '@/contexts/ConfigContext';
import { VAD_ENGINE_OPTIONS, type VadEngine } from '@/lib/vad';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

export function VadEngineSelect() {
  const { vadEngine, setVadEngine, vadEngineError } = useConfig();
  const current = VAD_ENGINE_OPTIONS.find(option => option.value === vadEngine);
  return (
    <div>
      <Label className="block text-sm font-medium text-gray-700 mb-1">Voice activity detection</Label>
      <div className="mx-1">
        <Select value={vadEngine} onValueChange={value => setVadEngine(value as VadEngine)}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {VAD_ENGINE_OPTIONS.map(option => (
              <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="mt-1 text-xs text-gray-500">
          {current?.description} Применяется к следующей записи, импорту и ретранскрибации.
        </p>
        {vadEngineError && <p role="alert" className="mt-1 text-xs text-red-600">{vadEngineError}</p>}
      </div>
    </div>
  );
}
