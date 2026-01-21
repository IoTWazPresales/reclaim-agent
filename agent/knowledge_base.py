"""Knowledge base generator for Reclaim codebase.

Analyzes the codebase structure, architecture, and patterns to create
a comprehensive knowledge base that the agent can use for context.

Uses hybrid approach:
- Python analysis (zero tokens): structure, file catalog, basic patterns
- LLM analysis (gpt-4o-mini, cheap): semantic understanding of key modules
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict


class KnowledgeBaseGenerator:
    """Generates a comprehensive knowledge base from the codebase."""
    
    def __init__(self, repo_path: str, openai_api_key: Optional[str] = None, use_llm_analysis: bool = True):
        self.repo_path = Path(repo_path)
        self.app_path = self.repo_path / "app"
        self.src_path = self.repo_path / "app" / "src"
        self.openai_api_key = openai_api_key
        self.use_llm_analysis = use_llm_analysis and openai_api_key is not None
        
    def generate(self) -> str:
        """Generate the complete knowledge base markdown."""
        sections = []
        
        # 1. Overview
        sections.append(self._generate_overview())
        
        # 2. Architecture
        sections.append(self._generate_architecture())
        
        # 3. Directory Structure
        sections.append(self._generate_directory_structure())
        
        # 4. Component Catalog (organized by module)
        sections.append(self._generate_component_catalog())
        
        # 5. Key Patterns and Conventions
        sections.append(self._generate_patterns())
        
        # 6. Import/Export Relationships
        sections.append(self._generate_imports_exports())
        
        # 7. Navigation Guide
        sections.append(self._generate_navigation_guide())
        
        # 8. Common Tasks
        sections.append(self._generate_common_tasks())
        
        # 9. Semantic Analysis (LLM-generated understanding of key modules)
        if self.use_llm_analysis:
            semantic_analysis = self._generate_semantic_analysis()
            if semantic_analysis:
                sections.append(semantic_analysis)
        
        return "\n\n".join(sections)
    
    def _generate_overview(self) -> str:
        """Generate overview section."""
        return """# Reclaim Application Knowledge Base

## Overview

Reclaim is a React Native application built with TypeScript, focusing on health and wellness tracking including:
- Training/workout management
- Medication tracking
- Sleep tracking
- Mood tracking
- Meditation/mindfulness
- Health data integration (Google Fit, Apple Health, Samsung Health)

## Technology Stack

- **Framework**: React Native
- **Language**: TypeScript
- **UI Library**: React Native Paper
- **State Management**: React Query (TanStack Query)
- **Testing**: Vitest
- **Build**: TypeScript compiler (tsc)

## Project Structure

```
app/
├── src/
│   ├── screens/          # Screen components (main UI pages)
│   ├── components/        # Reusable UI components
│   ├── lib/              # Core business logic and utilities
│   │   ├── training/     # Training/workout module
│   │   ├── health/       # Health integrations
│   │   ├── insights/     # Insight generation
│   │   └── ...           # Other modules
│   └── theme/            # Theming configuration
├── package.json
└── tsconfig.json
```
"""
    
    def _generate_architecture(self) -> str:
        """Generate architecture section."""
        return """## Architecture

### High-Level Architecture

1. **Screens Layer** (`app/src/screens/`)
   - Top-level UI pages
   - Handle navigation and user flow
   - Coordinate between components and lib modules

2. **Components Layer** (`app/src/components/`)
   - Reusable UI components
   - Organized by feature (training/, meditation/, dashboard/)
   - Presentational components with minimal business logic

3. **Library Layer** (`app/src/lib/`)
   - Core business logic
   - Data processing and transformations
   - API integrations
   - State management utilities

### Key Architectural Patterns

- **Separation of Concerns**: Screens handle UI flow, components handle presentation, lib handles logic
- **Module Organization**: Features grouped by domain (training, health, meditation, etc.)
- **Type Safety**: Extensive use of TypeScript types and interfaces
- **React Query**: Used for server state management and caching
- **Offline Support**: Queue-based offline sync for critical operations
"""
    
    def _generate_directory_structure(self) -> str:
        """Generate detailed directory structure."""
        if not self.src_path.exists():
            return "## Directory Structure\n\n(Unable to analyze - src path not found)"
        
        structure_parts = ["## Directory Structure\n"]
        structure_parts.append("```")
        structure_parts.append(self._build_tree(self.src_path, max_depth=4))
        structure_parts.append("```")
        
        return "\n".join(structure_parts)
    
    def _build_tree(self, path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
        """Build a tree representation of the directory structure."""
        if current_depth >= max_depth:
            return ""
        
        lines = []
        items = sorted([item for item in path.iterdir() if item.name != "node_modules" and not item.name.startswith(".")])
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            lines.append(f"{prefix}{current_prefix}{item.name}/" if item.is_dir() else f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(self._build_tree(item, next_prefix, max_depth, current_depth + 1))
        
        return "\n".join(lines)
    
    def _generate_component_catalog(self) -> str:
        """Generate component catalog organized by module."""
        catalog_parts = ["## Component Catalog\n"]
        catalog_parts.append("Components organized by module/feature area.\n")
        
        # Analyze screens
        screens = self._analyze_directory(self.src_path / "screens", "Screen")
        if screens:
            catalog_parts.append("### Screens (`app/src/screens/`)\n")
            catalog_parts.append("Main UI pages that users navigate to.\n")
            for screen in screens:
                catalog_parts.append(f"- **{screen['name']}**")
                catalog_parts.append(f"  - Path: `{screen['path']}`")
                if screen.get('exports'):
                    catalog_parts.append(f"  - Exports: {', '.join(screen['exports'])}")
                catalog_parts.append("")
        
        # Analyze components by subdirectory
        components_path = self.src_path / "components"
        if components_path.exists():
            catalog_parts.append("### Components (`app/src/components/`)\n")
            
            # Group by subdirectory
            component_groups = defaultdict(list)
            for comp in self._analyze_directory(components_path, "Component"):
                try:
                    comp_path_str = comp['path']
                    # Extract group from path string (more reliable than Path operations)
                    # Path format: app/src/components/<group>/<file> or app/src/components/<file>
                    if '/' in comp_path_str:
                        parts = comp_path_str.split('/')
                        # Find 'components' index
                        comp_idx = None
                        for i, part in enumerate(parts):
                            if part == 'components' and i + 1 < len(parts):
                                comp_idx = i + 1
                                break
                        if comp_idx is not None and comp_idx < len(parts) - 1:
                            # There's a subdirectory after components
                            group = parts[comp_idx]
                        else:
                            group = "root"
                    else:
                        group = "root"
                    component_groups[group].append(comp)
                except Exception as e:
                    # Debug logging (if needed, can add debug flag later)
                    pass
                    component_groups["root"].append(comp)
            
            for group, comps in sorted(component_groups.items()):
                if group != "root":
                    catalog_parts.append(f"#### {group.title()} Components\n")
                for comp in comps:
                    catalog_parts.append(f"- **{comp['name']}**")
                    catalog_parts.append(f"  - Path: `{comp['path']}`")
                    if comp.get('exports'):
                        catalog_parts.append(f"  - Exports: {', '.join(comp['exports'])}")
                    catalog_parts.append("")
        
        # Analyze lib modules
        lib_path = self.src_path / "lib"
        if lib_path.exists():
            catalog_parts.append("### Library Modules (`app/src/lib/`)\n")
            catalog_parts.append("Core business logic organized by domain.\n")
            
            # Group by subdirectory
            lib_groups = defaultdict(list)
            for module in self._analyze_directory(lib_path, "Module"):
                try:
                    mod_path = Path(module['path'])
                    if mod_path.is_absolute():
                        rel_path = mod_path.relative_to(lib_path.resolve())
                    else:
                        # Path is already relative, extract first part
                        rel_path = mod_path
                    group = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
                    lib_groups[group].append(module)
                except (ValueError, AttributeError):
                    # Fallback: extract group from path string
                    path_str = module['path']
                    if '/' in path_str:
                        # Extract lib/<group>/... from app/src/lib/<group>/...
                        parts = path_str.split('/')
                        lib_idx = None
                        for i, part in enumerate(parts):
                            if part == 'lib' and i + 1 < len(parts):
                                lib_idx = i + 1
                                break
                        group = parts[lib_idx] if lib_idx is not None else "root"
                    else:
                        group = "root"
                    lib_groups[group].append(module)
            
            for group, modules in sorted(lib_groups.items()):
                catalog_parts.append(f"#### {group.title()} Module\n")
                for module in modules:
                    catalog_parts.append(f"- **{module['name']}**")
                    catalog_parts.append(f"  - Path: `{module['path']}`")
                    if module.get('exports'):
                        catalog_parts.append(f"  - Exports: {', '.join(module['exports'][:5])}")  # Limit to first 5
                        if len(module['exports']) > 5:
                            catalog_parts.append(f"  - ... and {len(module['exports']) - 5} more")
                    if module.get('key_functions'):
                        catalog_parts.append(f"  - Key Functions: {', '.join(module['key_functions'][:3])}")
                    catalog_parts.append("")
        
        return "\n".join(catalog_parts)
    
    def _analyze_directory(self, path: Path, file_type: str) -> List[Dict[str, Any]]:
        """Analyze a directory and extract component/module information."""
        items = []
        
        if not path.exists():
            return items
        
        for file_path in path.rglob("*.tsx"):
            if file_path.name.endswith(".test.tsx") or file_path.name.endswith(".test.ts"):
                continue
            
            item = self._analyze_file(file_path, file_type)
            if item:
                items.append(item)
        
        # Also include .ts files for lib modules
        if "lib" in str(path):
            for file_path in path.rglob("*.ts"):
                if file_path.name.endswith(".test.ts"):
                    continue
                if file_path not in [Path(i['path']) for i in items]:
                    item = self._analyze_file(file_path, file_type)
                    if item:
                        items.append(item)
        
        return items
    
    def _analyze_file(self, file_path: Path, file_type: str) -> Optional[Dict[str, Any]]:
        """Analyze a single file and extract key information."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return None
        
        # Calculate relative path from repo root
        try:
            # Ensure both paths are absolute for relative_to() to work
            abs_file_path = file_path.resolve() if file_path.exists() else file_path
            if not abs_file_path.is_absolute():
                # If not absolute, make it relative to repo_path
                abs_file_path = (self.repo_path / abs_file_path).resolve()
            
            abs_repo_path = self.repo_path.resolve()
            
            # Try relative_to
            try:
                rel_path = abs_file_path.relative_to(abs_repo_path)
            except ValueError:
                # If relative_to fails, use os.path.relpath as fallback
                import os
                rel_path_str = os.path.relpath(str(abs_file_path), str(abs_repo_path))
                rel_path = Path(rel_path_str)
        except Exception as e:
            # Last resort: extract relative path from string representation
            file_str = str(file_path)
            # If it contains 'app/src/', extract everything after that
            if 'app/src/' in file_str:
                idx = file_str.find('app/src/')
                rel_path = Path(file_str[idx:])
            elif file_str.startswith('app/'):
                rel_path = Path(file_str)
            else:
                # Can't determine relative path, skip this file
                # Debug logging would go here if needed
                return None
        
        item = {
            'name': file_path.stem,
            'path': str(rel_path),
            'exports': [],
            'key_functions': []
        }
        
        # Extract exports
        export_patterns = [
            r'export\s+(?:default\s+)?(?:function|const|class|interface|type)\s+(\w+)',
            r'export\s+\{\s*([^}]+)\s*\}',
        ]
        
        for pattern in export_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, str):
                    # Handle named exports like `export { func1, func2 }`
                    exports = [e.strip() for e in match.split(',')]
                    item['exports'].extend(exports)
                else:
                    item['exports'].append(match)
        
        # Extract function definitions with more context
        func_patterns = [
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*[:=]\s*(?:async\s+)?\([^)]*\)\s*(?:[:=]|=>)',
            r'export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?:[:=]|=>)',
            r'export\s+(?:async\s+)?function\s+(\w+)\s*\(',
        ]
        for pattern in func_patterns:
            func_matches = re.findall(pattern, content)
            for func_name in func_matches:
                if func_name and func_name not in item['key_functions']:
                    item['key_functions'].append(func_name)
        
        # Remove duplicates
        item['exports'] = list(set(item['exports']))
        item['key_functions'] = list(set(item['key_functions']))[:10]  # Limit to 10
        
        return item
    
    def _generate_patterns(self) -> str:
        """Generate patterns and conventions section."""
        return """## Key Patterns and Conventions

### File Naming
- **Screens**: `*Screen.tsx` (e.g., `TrainingScreen.tsx`)
- **Components**: PascalCase (e.g., `ExerciseCard.tsx`)
- **Utilities**: camelCase (e.g., `circadianUtils.ts`)
- **Types**: `types.ts` or `*.types.ts`

### Import Patterns
- Use `@/` alias for imports from `app/src/` (configured in tsconfig.json)
- Example: `import { logger } from '@/lib/logger'`
- Group imports: external → internal → relative

### Component Patterns
- Functional components with hooks
- React Query for data fetching
- React Native Paper for UI components
- TypeScript interfaces for props

### State Management
- **Server State**: React Query (`useQuery`, `useMutation`)
- **Local State**: React hooks (`useState`, `useReducer`)
- **Global State**: Context API (when needed)

### Error Handling
- Try-catch blocks for async operations
- Error logging via `@/lib/logger`
- User-friendly error messages via React Native Paper `Alert`

### Testing
- Vitest for unit tests
- Test files: `*.test.ts` or `*.test.tsx`
- Located in `__tests__/` directories or alongside source files
"""
    
    def _generate_imports_exports(self) -> str:
        """Generate import/export relationships."""
        return """## Import/Export Relationships

### Key Entry Points

#### Training Module
- **Main Engine**: `app/src/lib/training/engine/index.ts`
  - Exports: `buildSessionFromProgramDay`, `summarizeSessionPlan`
- **Types**: `app/src/lib/training/types.ts`
  - Exports: `TrainingGoal`, `TrainingSession`, `TrainingGoalSettings`, etc.
- **Program Planner**: `app/src/lib/training/programPlanner.ts`
  - Exports: `buildFourWeekPlan`, `generateProgramDays`

#### Health Integrations
- **Main Index**: `app/src/lib/health/index.ts`
  - Exports: Health integration utilities
- **Providers**: `app/src/lib/health/providers/`
  - Google Fit, Apple Health, Samsung Health implementations

#### UI Components
- **Training Components**: `app/src/components/training/`
  - `ExerciseCard`, `FullSessionPanel`, `FourWeekPreview`, etc.
- **Common Components**: `app/src/components/`
  - `InsightCard`, `ProgressRing`, `CalendarCard`, etc.

### Common Import Paths
- `@/lib/training/engine` - Training session generation
- `@/lib/training/types` - Training type definitions
- `@/lib/training/programPlanner` - Program planning logic
- `@/lib/logger` - Logging utilities
- `@/theme` - Theme configuration
- `react-native-paper` - UI components
- `@tanstack/react-query` - Data fetching
"""
    
    def _generate_navigation_guide(self) -> str:
        """Generate navigation guide for common tasks."""
        return """## Navigation Guide

### How to Find Things

#### Finding a Screen
1. Check `app/src/screens/` for top-level screens
2. Check `app/src/screens/<feature>/` for feature-specific screens
3. Example: Training screens → `app/src/screens/training/`

#### Finding a Component
1. Check `app/src/components/` for common components
2. Check `app/src/components/<feature>/` for feature-specific components
3. Example: Training components → `app/src/components/training/`

#### Finding Business Logic
1. Check `app/src/lib/<feature>/` for feature-specific logic
2. Example: Training logic → `app/src/lib/training/`
3. Look for `index.ts` files for main exports

#### Finding Types/Interfaces
1. Check `app/src/lib/<feature>/types.ts`
2. Check `app/src/lib/<feature>/<module>.ts` for inline types
3. Example: Training types → `app/src/lib/training/types.ts`

#### Finding API/Data Layer
1. Check `app/src/lib/api.ts` for API utilities
2. Check `app/src/lib/<feature>/` for feature-specific data handling
3. Look for files with `Service`, `Store`, or `Queue` in the name

### Common File Locations

- **Training Setup Screen**: `app/src/screens/training/TrainingSetupScreen.tsx`
- **Training Engine**: `app/src/lib/training/engine/index.ts`
- **Training Types**: `app/src/lib/training/types.ts`
- **Training Components**: `app/src/components/training/`
- **Health Integrations**: `app/src/lib/health/`
- **Insights**: `app/src/lib/insights/`
- **Theme**: `app/src/theme/` (if exists)
"""
    
    def _generate_common_tasks(self) -> str:
        """Generate common tasks guide."""
        return """## Common Tasks and How to Accomplish Them

### Adding a New Screen
1. Create file in `app/src/screens/` or `app/src/screens/<feature>/`
2. Import React Native Paper components
3. Use React Query for data fetching if needed
4. Add navigation route (if applicable)

### Adding a New Component
1. Create file in `app/src/components/` or `app/src/components/<feature>/`
2. Define TypeScript interface for props
3. Use React Native Paper for UI
4. Export component

### Adding Training Functionality
1. Business logic → `app/src/lib/training/`
2. UI components → `app/src/components/training/`
3. Screens → `app/src/screens/training/`
4. Types → `app/src/lib/training/types.ts`

### Modifying Training Engine
1. Main engine: `app/src/lib/training/engine/index.ts`
2. Session generation: `buildSessionFromProgramDay` function
3. Program planning: `app/src/lib/training/programPlanner.ts`
4. Types: `app/src/lib/training/types.ts`

### Adding Health Integration
1. Provider implementation → `app/src/lib/health/providers/`
2. Service layer → `app/src/lib/health/`
3. Types → `app/src/lib/health/types.ts`
4. Register in `app/src/lib/health/index.ts`

### Testing Changes
1. Run: `cd app && npx tsc --noEmit` (TypeScript check)
2. Run: `cd app && npx vitest run --passWithNoTests` (Unit tests)
3. Fix any type errors or test failures
"""
    
    def _generate_semantic_analysis(self) -> Optional[str]:
        """Generate semantic analysis of key modules using LLM (gpt-4o-mini)."""
        if not self.use_llm_analysis:
            return None
        
        # Key modules to analyze semantically
        key_modules = [
            {
                "name": "Training Module",
                "path": "app/src/lib/training",
                "files": [
                    "app/src/lib/training/engine/index.ts",
                    "app/src/lib/training/types.ts",
                    "app/src/lib/training/programPlanner.ts",
                    "app/src/lib/training/setupMappings.ts",
                ],
                "description": "Core training/workout generation logic"
            },
            {
                "name": "Training Screens",
                "path": "app/src/screens/training",
                "files": [
                    "app/src/screens/training/TrainingSetupScreen.tsx",
                    "app/src/screens/training/TrainingScreen.tsx",
                ],
                "description": "Training UI screens and user flows"
            },
            {
                "name": "Training Components",
                "path": "app/src/components/training",
                "files": [
                    "app/src/components/training/FullSessionPanel.tsx",
                    "app/src/components/training/FourWeekPreview.tsx",
                ],
                "description": "Training UI components"
            },
        ]
        
        semantic_parts = ["## Semantic Analysis (LLM-Generated Understanding)\n"]
        semantic_parts.append("This section provides semantic understanding of key modules, generated using LLM analysis (gpt-4o-mini).\n")
        
        for module in key_modules:
            analysis = self._analyze_module_semantically(module)
            if analysis:
                semantic_parts.append(analysis)
                semantic_parts.append("")  # Blank line between modules
        
        return "\n".join(semantic_parts) if len(semantic_parts) > 2 else None
    
    def _analyze_module_semantically(self, module: Dict[str, Any]) -> Optional[str]:
        """Analyze a module semantically using LLM."""
        try:
            # Read key files from the module
            file_contents = []
            for file_path_str in module.get("files", []):
                file_path = self.repo_path / file_path_str
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        # Limit file size to avoid token bloat (first 3000 lines or 100k chars)
                        lines = content.split("\n")
                        if len(lines) > 3000:
                            content = "\n".join(lines[:3000]) + "\n... [truncated for analysis]"
                        elif len(content) > 100000:
                            content = content[:100000] + "\n... [truncated for analysis]"
                        file_contents.append(f"### {file_path_str}\n```typescript\n{content}\n```")
                    except Exception as e:
                        if self._debug_enabled():
                            print(f"[KB] Failed to read {file_path_str}: {e}")
            
            if not file_contents:
                return None
            
            # Build prompt for semantic analysis
            prompt = f"""Analyze the following {module['name']} code and provide a comprehensive understanding:

{module.get('description', '')}

Files:
{chr(10).join(file_contents[:3])}  # Limit to 3 files to control token usage

Provide:
1. **Purpose**: What does this module do? What problem does it solve?
2. **Key Functions**: What are the main functions and what do they do?
3. **Data Flow**: How does data flow through this module?
4. **Dependencies**: What other modules/files does it depend on?
5. **Usage Patterns**: How is this module typically used by other parts of the codebase?
6. **Important Types/Interfaces**: What are the key types and their purposes?
7. **Business Logic**: What are the core business rules/logic implemented here?

Format as clear, structured markdown. Be specific and reference actual function names, types, and patterns from the code.
"""
            
            # Call LLM (gpt-4o-mini for cost efficiency)
            response = self._call_llm_for_analysis(prompt)
            if response:
                return f"### {module['name']}\n\n{response}"
            
        except Exception as e:
            if self._debug_enabled():
                print(f"[KB] Semantic analysis failed for {module['name']}: {e}")
        
        return None
    
    def _call_llm_for_analysis(self, prompt: str) -> Optional[str]:
        """Call OpenAI API (gpt-4o-mini) for semantic analysis."""
        if not self.openai_api_key:
            return None
        
        try:
            import requests
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": "gpt-4o-mini",  # Cheap model for analysis
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code analysis assistant. Analyze code and provide clear, structured documentation of its purpose, functions, data flow, and business logic."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "max_tokens": 2000,  # Limit output to control costs
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if self._debug_enabled():
                usage = result.get("usage", {})
                print(f"[KB] LLM analysis: {usage.get('prompt_tokens', 0)} input, {usage.get('completion_tokens', 0)} output tokens")
            
            return content.strip() if content else None
            
        except Exception as e:
            if self._debug_enabled():
                print(f"[KB] LLM call failed: {e}")
            return None
    
    def _debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("AGENT_DEBUG", "").strip().lower() in ("1", "true", "yes")
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Generate and save knowledge base to file."""
        if output_path is None:
            output_path = self.repo_path / "KNOWLEDGE_BASE.md"
        
        kb_content = self.generate()
        output_path.write_text(kb_content, encoding="utf-8")
        
        return output_path
