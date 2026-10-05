"""Generate dependency-free CLI and source API references from this checkout."""
import argparse
import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate():
    """Write includes used by Sphinx; never import numerical runtime modules."""
    from openspliceai.openspliceai import build_parser
    destination = ROOT/'docs/source/_generated'
    destination.mkdir(exist_ok=True)
    parser = build_parser()
    parser.prog = 'openspliceai'
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    for name, command in subcommands.choices.items():
        command.prog = f'openspliceai {name}'
        text = ['.. code-block:: text', '', *('   '+line for line in command.format_usage().splitlines()), '',
                '.. list-table:: Options', '   :header-rows: 1', '   :widths: 25 15 60', '',
                '   * - Option', '     - Default', '     - Meaning']
        for action in command._actions:
            if action.dest == 'help':
                continue
            default = 'required' if action.required else str(action.default)
            if action.dest == 'input_vcf':
                default = 'stdin'
            if action.dest == 'output_vcf':
                default = 'stdout'
            choices = ''
            if action.choices is not None:
                choices = (' Choices: '+', '.join(map(str, action.choices))+'.' if len(action.choices) < 20
                           else f' Choices: {min(action.choices)} through {max(action.choices)}.')
            help_text = (action.help or 'Experiment model label.').replace('%(default)s', str(action.default))
            help_text = re.sub(r'\s+', ' ', help_text+choices).replace('*', r'\*').replace('_', r'\_')
            text += ['   * - ``'+', '.join(action.option_strings)+'``', '     - ``'+default+'``', '     - '+help_text]
        (destination/f'cli-{name}.inc').write_text('\n'.join(text)+'\n')
    text = ['Generated from public definitions in the installed package. Lower-level helpers',
            'are included for source navigation; the six command entrypoints define the supported',
            'end-to-end workflows. Research tools are documented separately.', '']
    excluded = {'scripts', 'test'}
    for path in sorted((ROOT/'openspliceai').rglob('*.py')):
        if excluded.intersection(path.relative_to(ROOT/'openspliceai').parts):
            continue
        if path.name in ('temperature_scaling_site_only.py', 'gff_to_tsv.py'):
            continue
        tree = ast.parse(path.read_text())
        entries = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                entries.append((node.name+'('+ast.unparse(node.args)+')', ast.get_docstring(node), node.lineno))
            if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                entries.append((node.name, ast.get_docstring(node), node.lineno))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not child.name.startswith('_') or child.name in ('__init__', '__enter__', '__exit__')):
                        entries.append((node.name+'.'+child.name+'('+ast.unparse(child.args)+')', ast.get_docstring(child), child.lineno))
        if not entries:
            continue
        module = str(path.relative_to(ROOT)).replace('/', '.')[:-3]
        text += [module, '~'*len(module), '']
        for signature, docstring, line in entries:
            url = 'https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/audit/comprehensive-20261004/'+str(path.relative_to(ROOT))+f'#L{line}'
            text += ['``'+signature.replace('`', '')+'``', '   `Source <'+url+'>`__.', '']
            if docstring:
                summary = docstring.splitlines()[0]
                # Literal summaries avoid accidental RST markup in legacy docstrings.
                text += ['   .. code-block:: text', '', '      '+summary, '']
    (destination/'api.inc').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    generate()
