"use client";

import { ModelConfig, ModelSettingsModal } from '@/components/ModelSettingsModal';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle as DialogTitleVisible,
  DialogTrigger,
  DialogTitle,
} from "@/components/ui/dialog"
import { VisuallyHidden } from "@/components/ui/visually-hidden"
import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, Settings, Loader2, FileText, Check, Square, Plus, Pencil, Trash2 } from 'lucide-react';
import Analytics from '@/lib/analytics';
import { toast } from 'sonner';
import { useState, useEffect, ReactNode } from 'react';

interface SummaryGeneratorButtonGroupProps {
  languageSlot?: ReactNode;
  modelConfig: ModelConfig;
  setModelConfig: (config: ModelConfig | ((prev: ModelConfig) => ModelConfig)) => void;
  onSaveModelConfig: (config?: ModelConfig) => Promise<void>;
  onGenerateSummary: (customPrompt: string) => Promise<void>;
  onStopGeneration: () => void;
  customPrompt: string;
  summaryStatus: 'idle' | 'processing' | 'summarizing' | 'regenerating' | 'completed' | 'error';
  availableTemplates: Array<{ id: string, name: string, description: string, is_custom: boolean }>;
  selectedTemplate: string;
  onTemplateSelect: (templateId: string, templateName: string) => void;
  onGetTemplateJson: (templateId: string) => Promise<string | null>;
  onSaveTemplate: (templateId: string, templateJson: string) => Promise<boolean>;
  onDeleteTemplate: (templateId: string) => Promise<boolean>;
  hasTranscripts?: boolean;
  hasSummary?: boolean;
  isModelConfigLoading?: boolean;
  onOpenModelSettings?: (openFn: () => void) => void;
}

export function SummaryGeneratorButtonGroup({
  modelConfig,
  setModelConfig,
  onSaveModelConfig,
  onGenerateSummary,
  onStopGeneration,
  customPrompt,
  summaryStatus,
  availableTemplates,
  selectedTemplate,
  onTemplateSelect,
  onGetTemplateJson,
  onSaveTemplate,
  onDeleteTemplate,
  hasTranscripts = true,
  hasSummary = false,
  isModelConfigLoading = false,
  onOpenModelSettings,
  languageSlot
}: SummaryGeneratorButtonGroupProps) {
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [templateDialogMode, setTemplateDialogMode] = useState<'create' | 'edit'>('create');
  const [templateIdInput, setTemplateIdInput] = useState('');
  const [templateJsonInput, setTemplateJsonInput] = useState('');
  const [isTemplateSaving, setIsTemplateSaving] = useState(false);
  const defaultTemplateSkeleton = `{
  "name": "Новый шаблон",
  "description": "Описание шаблона",
  "sections": [
    {
      "title": "Краткое резюме",
      "instruction": "Кратко перечисли ключевые пункты встречи по фактам.",
      "format": "list"
    }
  ]
}`;

  // Expose the function to open the modal via callback registration
  useEffect(() => {
    if (onOpenModelSettings) {
      // Register our open dialog function with the parent by calling the callback
      // This allows the parent to store a reference to this function
      const openDialog = () => {
        console.log('📱 Opening model settings dialog via callback');
        setSettingsDialogOpen(true);
      };

      // Call the parent's callback with our open function
      // Note: This assumes onOpenModelSettings accepts a function parameter
      // We'll need to adjust the signature
      onOpenModelSettings(openDialog);
    }
  }, [onOpenModelSettings]);

  if (!hasTranscripts) {
    return null;
  }

  const openCreateTemplateDialog = () => {
    setTemplateDialogMode('create');
    setTemplateIdInput('');
    setTemplateJsonInput(defaultTemplateSkeleton);
    setTemplateDialogOpen(true);
  };

  const openEditTemplateDialog = async (templateId: string) => {
    const templateJson = await onGetTemplateJson(templateId);
    if (!templateJson) {
      return;
    }

    setTemplateDialogMode('edit');
    setTemplateIdInput(templateId);
    setTemplateJsonInput(templateJson);
    setTemplateDialogOpen(true);
  };

  const handleSaveTemplate = async () => {
    const templateId = templateIdInput.trim();
    if (!templateId) {
      toast.error('Template id is required');
      return;
    }
    if (!templateJsonInput.trim()) {
      toast.error('Template JSON is required');
      return;
    }

    setIsTemplateSaving(true);
    const saved = await onSaveTemplate(templateId, templateJsonInput);
    setIsTemplateSaving(false);

    if (saved) {
      setTemplateDialogOpen(false);
    }
  };

  const handleDeleteTemplate = async (templateId: string, templateName: string) => {
    const confirmed = window.confirm(`Удалить шаблон "${templateName}"?`);
    if (!confirmed) {
      return;
    }
    await onDeleteTemplate(templateId);
  };

  const isGenerating = summaryStatus === 'processing' || summaryStatus === 'summarizing' || summaryStatus === 'regenerating';

  return (
    <ButtonGroup>
      {/* Generate Summary or Stop button */}
      {isGenerating ? (
        <Button
          variant="outline"
          size="sm"
          className="bg-gradient-to-r from-red-50 to-orange-50 hover:from-red-100 hover:to-orange-100 border-red-200 px-3 gap-2"
          onClick={() => {
            Analytics.trackButtonClick('stop_summary_generation', 'meeting_details');
            onStopGeneration();
          }}
          title="Stop summary generation"
        >
          <Square size={18} fill="currentColor" />
          <span className="hidden @[24rem]:inline">Stop</span>
        </Button>
      ) : (
        <Button
          variant="outline"
          size="sm"
          className="bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border-blue-200 px-3 gap-2"
          onClick={() => {
            Analytics.trackButtonClick('generate_summary', 'meeting_details');
            void onGenerateSummary(customPrompt);
          }}
          disabled={isModelConfigLoading}
          title={
            isModelConfigLoading
              ? 'Loading model configuration...'
              : hasSummary ? 'Regenerate AI Summary' : 'Generate AI Summary'
          }
        >
          {isModelConfigLoading ? (
            <>
              <Loader2 className="animate-spin" size={18} />
              <span className="hidden @[24rem]:inline">Processing...</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span className="hidden @[24rem]:inline">{hasSummary ? 'Regenerate Summary' : 'Generate Summary'}</span>
            </>
          )}
        </Button>
      )}

      {languageSlot}

      {/* Settings button */}
      <Dialog open={settingsDialogOpen} onOpenChange={setSettingsDialogOpen}>
        <DialogTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            title="Summary Settings"
          >
            <Settings />
            <span className="hidden @[40rem]:inline">AI Model</span>
          </Button>
        </DialogTrigger>
        <DialogContent
          aria-describedby={undefined}
        >
          <VisuallyHidden>
            <DialogTitle>Model Settings</DialogTitle>
          </VisuallyHidden>
          <ModelSettingsModal
            onSave={async (config) => {
              await onSaveModelConfig(config);
              setSettingsDialogOpen(false);
            }}
            modelConfig={modelConfig}
            setModelConfig={setModelConfig}
            skipInitialFetch={true}
            layout="dialog"
          />
        </DialogContent>
      </Dialog>

      {/* Template editor dialog */}
      <Dialog open={templateDialogOpen} onOpenChange={setTemplateDialogOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitleVisible>
              {templateDialogMode === 'create' ? 'Добавить шаблон' : 'Редактировать шаблон'}
            </DialogTitleVisible>
            <DialogDescription>
              Шаблон сохраняется как JSON и валидируется перед сохранением.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-3">
            <label className="text-sm font-medium">Template ID</label>
            <Input
              value={templateIdInput}
              onChange={(e) => setTemplateIdInput(e.target.value)}
              placeholder="example: russian_meeting_protocol"
              disabled={templateDialogMode === 'edit'}
            />
            <p className="text-xs text-muted-foreground">
              Используйте только буквы, цифры, `_` и `-`.
            </p>
          </div>

          <div className="grid gap-3">
            <label className="text-sm font-medium">Template JSON</label>
            <Textarea
              value={templateJsonInput}
              onChange={(e) => setTemplateJsonInput(e.target.value)}
              className="min-h-[360px] font-mono text-xs"
              placeholder='{"name":"...","description":"...","sections":[...]}'
            />
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setTemplateDialogOpen(false)}>
              Отмена
            </Button>
            <Button onClick={handleSaveTemplate} disabled={isTemplateSaving}>
              {isTemplateSaving ? 'Сохранение...' : 'Сохранить'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Template selector dropdown */}
      {availableTemplates.length > 0 && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              title="Select summary template"
            >
              <FileText />
              <span className="hidden @[40rem]:inline">Template</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {availableTemplates.map((template) => (
              <DropdownMenuItem
                key={template.id}
                onClick={() => onTemplateSelect(template.id, template.name)}
                title={template.description}
                className="flex items-center justify-between gap-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate">{template.name}</div>
                  <div className="truncate text-xs text-muted-foreground">{template.description}</div>
                </div>
                <div className="flex items-center gap-1">
                  {selectedTemplate === template.id && (
                    <Check className="h-4 w-4 text-green-600" />
                  )}
                  <button
                    type="button"
                    className="inline-flex h-7 w-7 items-center justify-center rounded hover:bg-accent"
                    title="Редактировать шаблон"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      openEditTemplateDialog(template.id);
                    }}
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    className="inline-flex h-7 w-7 items-center justify-center rounded hover:bg-accent"
                    title={template.is_custom ? 'Удалить шаблон' : 'Удалить пользовательский override'}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      handleDeleteTemplate(template.id, template.name);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={openCreateTemplateDialog}>
              <Plus className="h-4 w-4" />
              <span>Добавить шаблон</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </ButtonGroup>
  );
}
