import { paletteItems } from '../../data/paletteItems';

export function Palette() {
  const categories = [...new Set(paletteItems.map((item) => item.category))];
  return (
    <aside className="palette">
      <div className="panel-heading">
        <strong>组件库</strong>
        <span>拖入画布</span>
      </div>
      <div className="palette-scroll">
        {categories.map((category) => (
          <section key={category}>
            <h3>{category}</h3>
            {paletteItems
              .filter((item) => item.category === category)
              .map((item) => (
                <div
                  key={item.nodeType}
                  className={`palette-item kind-${item.nodeType}`}
                  draggable
                  title={item.description}
                  onDragStart={(event) => {
                    event.dataTransfer.setData('application/watermet-node', item.nodeType);
                    event.dataTransfer.effectAllowed = 'move';
                  }}
                >
                  <span className="palette-dot" />
                  <div>
                    <strong>{item.label}</strong>
                    <small>{item.description}</small>
                  </div>
                </div>
              ))}
          </section>
        ))}
      </div>
    </aside>
  );
}
