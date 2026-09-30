from pathlib import Path

import nbformat


root = Path(__file__).resolve().parents[1]
source = (root / "src" / "experiment.py").read_text(encoding="utf-8")
source = source[:source.index('if __name__ == "__main__":')]
source = source.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
sections = [
    ("# Transfer learning y fine-tuning en CIFAR-10\n\n"
     "El cuaderno ejecuta nueve configuraciones sobre 45 000 imágenes, escoge modelos exclusivamente con validación, "
     "y compara los seleccionados con un nuevo entrenamiento sobre 4 500 imágenes. "
     "Se usa la semilla 42 y la partición oficial de 10 000 imágenes se reserva para test. "
     "Las salidas se generan al ejecutar las celdas en orden desde la raíz del proyecto."),
    ("## Investigación de PyTorch\n\n"
     "`vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` carga los pesos preentrenados. "
     "`VGG16_Weights.IMAGENET1K_V1.transforms()` ofrece reescalado, recorte y normalización de referencia. "
     "Se conserva la normalización ImageNet, pero se redimensiona a 112×112 en ambas VGG para limitar el cómputo. "
     "`features` contiene cinco bloques convolucionales, `avgpool` es `nn.AdaptiveAvgPool2d((7, 7))` "
     "y `classifier` contiene las capas densas. Se sustituye `classifier[6]` por una salida de diez clases.\n\n"
     "`requires_grad=False` congela los parámetros del backbone; con un bloque o dos descongelados, "
     "los grupos de parámetros del optimizador asignan `backbone_lr` a estos y `lr` al clasificador. "
     "`model.train()` activa dropout y la actualización de batch normalization; `model.eval()` "
     "desactiva dropout y usa las estadísticas aprendidas. En evaluación, `torch.set_grad_enabled(False)` "
     "tiene el mismo propósito de no guardar el grafo que `torch.no_grad()`.\n\n"
     "`torch.profiler.profile(with_flops=True)` estima operaciones por imagen para convoluciones y matrices. "
     "Un MAC reúne una multiplicación y una suma, aproximadamente dos FLOPs; no todas las operaciones están cubiertas. "
     "Los parámetros totales incluyen los congelados, mientras que los entrenables tienen `requires_grad=True`. "
     "`torch.cuda.synchronize()` termina las operaciones GPU pendientes antes de medir tiempo; "
     "`torch.cuda.max_memory_allocated()` lee el pico de memoria de tensores desde el último reinicio del contador."),
    ("## Datos y preparación\n\n"
     "CIFAR-10 contiene 50 000 imágenes originales de entrenamiento y 10 000 de test, de 32×32 píxeles RGB. "
     "Cada una de las diez clases tiene 5 000 ejemplos de entrenamiento. Se separan 45 000 para entrenar "
     "y 5 000 para validar, estratificados por clase. La media y desviación estándar por canal se calculan "
     "solo sobre entrenamiento y se imprimen junto a las de ImageNet. Gato/perro, automóvil/camión "
     "y ciervo/caballo pueden confundirse por sus siluetas y su baja resolución.\n\n"
     "La CNN conserva 32×32 y normaliza con estadísticas de CIFAR-10. VGG usa 112×112 y normalización ImageNet. "
     "Pasar de 32 a 112 multiplica por 12,25 el número de píxeles; usar 224 lo multiplicaría por 49. "
     "El aumento de datos usa recorte aleatorio con relleno y reflejo horizontal para variar posiciones y orientación. "
     "Solo se aplica en entrenamiento, ya que validación y test deben medir imágenes originales sin cambios aleatorios."),
    ("## Entrenamiento\n\n"
     "La CNN propia contiene tres bloques convolucionales con batch normalization y dropout. "
     "La extracción congela los cinco bloques VGG; el fine-tuning compara uno y dos bloques descongelados "
     "con una tasa de aprendizaje menor para el backbone. Cada configuración registra pérdida y accuracy, "
     "precision, recall y F1 macro en cada época, además de tiempo y memoria. "
     "La mejor época se conserva por F1 macro de validación. El test no participa en la selección."),
    ("## Evaluación final y recursos\n\n"
     "Solo las tres configuraciones seleccionadas se evalúan sobre test. Luego se entrenan nuevamente desde cero "
     "para la CNN o desde ImageNet para VGG, con un 10 % estratificado del conjunto de entrenamiento. "
     "Para el costo estimado se usa, por imagen y por todas las épocas efectivamente entrenadas, "
     "un forward de la parte congelada y tres veces el forward de la parte entrenable. "
     "Esta cifra y los FLOPs del profiler son aproximaciones, no un recuento exacto de todas las operaciones."),
]
notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_markdown_cell(sections[0]),
    nbformat.v4.new_markdown_cell(sections[1]),
    nbformat.v4.new_markdown_cell(sections[2]),
    nbformat.v4.new_code_cell(source),
    nbformat.v4.new_code_cell("train, val, reduced, classes = prepare()\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(RESULTS / 'examples.png')))\nprint((RESULTS / 'dataset.json').read_text(encoding='utf-8'))"),
    nbformat.v4.new_markdown_cell(sections[3]),
    nbformat.v4.new_code_cell("for config in CONFIGS:\n    run(config, train, val, classes)"),
    nbformat.v4.new_markdown_cell(sections[4]),
    nbformat.v4.new_code_cell("winners = {}\nfor kind in ('cnn', 'extract', 'tune'):\n    candidates = [json.loads((RESULTS / f\"{c['name']}.json\").read_text(encoding='utf-8')) for c in CONFIGS if c['kind'] == kind]\n    winner = max(candidates, key=lambda item: item['best_validation']['f1'])\n    winners[kind] = winner['config']['name']\n    evaluate_selected(winner['config'], classes)\n    run(winner['config'], reduced, val, classes, reduced=True)\n    evaluate_selected(winner['config'], classes, reduced=True)\n(RESULTS / 'winners.json').write_text(json.dumps(winners, indent=2), encoding='utf-8')\nplot_results()\nprint(winners)"),
    nbformat.v4.new_code_cell(
        "print('Iteración | F1 validación | Época | Parámetros totales / entrenables | Tiempo total (s) | Pico GPU (GiB)')\n"
        "for config in CONFIGS:\n"
        "    result = json.loads((RESULTS / f\"{config['name']}.json\").read_text(encoding='utf-8'))\n"
        "    peak = result['peak_gpu_bytes'] / 2**30 if result['peak_gpu_bytes'] else 0\n"
        "    print(f\"{config['name']:10s} {result['best_validation']['f1']:.4f}  {result['selected_epoch']}  \"\n"
        "          f\"{result['total_params']:,} / {result['trainable_params']:,}  \"\n"
        "          f\"{result['total_seconds']:.1f}  {peak:.2f}\")\n"
        "print('\\nGanadores: métricas de test y recursos')\n"
        "for kind, name in winners.items():\n"
        "    full = json.loads((RESULTS / f'{name}.json').read_text(encoding='utf-8'))\n"
        "    small = json.loads((RESULTS / f'{name}_10pct.json').read_text(encoding='utf-8'))\n"
        "    print(kind, name, 'test:', full['test'], 'test 10 %:', small['test'])\n"
        "    print('FLOPs/imagen:', full['inference_flops'], 'latencia ms:', full['inference_ms'],\n"
        "          'FLOPs entrenamiento:', full['training_flops_estimate'])"
    ),
    nbformat.v4.new_code_cell("for filename in ('loss_cnn.png', 'loss_extract.png', 'loss_tune.png', 'accuracy_vs_time.png', 'f1_vs_flops.png', 'confusions.png'):\n    display(Image(filename=str(RESULTS / filename)))"),
]
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata["language_info"] = {"name": "python"}
(root / "notebooks").mkdir(exist_ok=True)
nbformat.write(notebook, root / "notebooks" / "transfer_learning.ipynb")
