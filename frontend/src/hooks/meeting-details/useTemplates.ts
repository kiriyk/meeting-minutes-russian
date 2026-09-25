import { useState, useEffect, useCallback } from 'react';
import { invoke as invokeTauri } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import Analytics from '@/lib/analytics';

type TemplateInfo = {
  id: string;
  name: string;
  description: string;
  is_custom: boolean;
};

export function useTemplates() {
  const [availableTemplates, setAvailableTemplates] = useState<TemplateInfo[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('standard_meeting');

  const fetchTemplates = useCallback(async (): Promise<TemplateInfo[]> => {
    try {
      const templates = await invokeTauri('api_list_templates') as TemplateInfo[];
      console.log('Available templates:', templates);
      setAvailableTemplates(templates);
      return templates;
    } catch (error) {
      console.error('Failed to fetch templates:', error);
      return [];
    }
  }, []);

  // Fetch available templates on mount
  useEffect(() => {
    fetchTemplates();
  }, [fetchTemplates]);

  // Keep selection valid if templates list changes
  useEffect(() => {
    if (availableTemplates.length === 0) {
      return;
    }

    const selectedExists = availableTemplates.some((template) => template.id === selectedTemplate);
    if (selectedExists) {
      return;
    }

    const fallback = availableTemplates.find((template) => template.id === 'standard_meeting')
      ?? availableTemplates[0];
    setSelectedTemplate(fallback.id);
  }, [availableTemplates, selectedTemplate]);

  // Handle template selection
  const handleTemplateSelection = useCallback((templateId: string, templateName: string) => {
    setSelectedTemplate(templateId);
    toast.success('Template selected', {
      description: `Using "${templateName}" template for summary generation`,
    });
    Analytics.trackFeatureUsed('template_selected');
  }, []);

  const getTemplateJson = useCallback(async (templateId: string): Promise<string | null> => {
    try {
      return await invokeTauri('api_get_template_json', { templateId }) as string;
    } catch (error) {
      console.error('Failed to load template JSON:', error);
      toast.error('Failed to load template');
      return null;
    }
  }, []);

  const saveTemplate = useCallback(async (
    templateId: string,
    templateJson: string
  ): Promise<boolean> => {
    try {
      // Explicit validation call for immediate feedback before save
      await invokeTauri('api_validate_template', { templateJson });
      const savedTemplateName = await invokeTauri('api_save_template', {
        templateId,
        templateJson,
      }) as string;

      await fetchTemplates();
      setSelectedTemplate(templateId);

      toast.success('Template saved', {
        description: `"${savedTemplateName}" is ready to use`,
      });
      Analytics.trackFeatureUsed('template_saved');
      return true;
    } catch (error) {
      console.error('Failed to save template:', error);
      toast.error('Template validation or save failed', {
        description: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }, [fetchTemplates]);

  const deleteTemplate = useCallback(async (templateId: string): Promise<boolean> => {
    try {
      await invokeTauri('api_delete_template', { templateId });
      const updatedTemplates = await fetchTemplates();

      if (selectedTemplate === templateId && updatedTemplates.length > 0) {
        const fallback = updatedTemplates.find((template) => template.id === 'standard_meeting')
          ?? updatedTemplates[0];
        setSelectedTemplate(fallback.id);
      }

      toast.success('Template deleted');
      Analytics.trackFeatureUsed('template_deleted');
      return true;
    } catch (error) {
      console.error('Failed to delete template:', error);
      toast.error('Failed to delete template', {
        description: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }, [fetchTemplates, selectedTemplate]);

  return {
    availableTemplates,
    selectedTemplate,
    handleTemplateSelection,
    getTemplateJson,
    saveTemplate,
    deleteTemplate,
    refreshTemplates: fetchTemplates,
  };
}
