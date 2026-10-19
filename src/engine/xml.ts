/**
 * A scanner for OpenMM's XML serialisation.
 *
 * Not a general XML parser.  OpenMM's `XmlSerializer` output is flat, regular,
 * attribute-only and machine-written: no text content, no namespaces, no
 * entities beyond what attribute values need, and no mixed content.  A scanner
 * that handles exactly that is a few dozen lines, has no dependency, runs the
 * same in Node and the browser, and allocates nothing per element -- which
 * matters, because a solvated system's `system.xml` is 7.3 MB and holds about
 * 75000 elements (26078 particles, 37732 exceptions, 8894 torsions).
 *
 * `DOMParser` would do this in the browser but does not exist in Node, and the
 * alternative of building 75000 objects to read six numbers out of each is
 * what this avoids.
 */

export class XmlError extends Error {
  constructor(message: string, readonly offset: number) {
    super(`${message} (at character ${offset})`);
    this.name = "XmlError";
  }
}

export class XmlScanner {
  private index = 0;
  /** Name of the tag the scanner is currently on. */
  name = "";
  /** True when the current tag is a closing tag, `</Foo>`. */
  closing = false;
  /** True when the current tag closes itself, `<Foo/>`. */
  selfClosing = false;
  private attributeStart = 0;
  private attributeEnd = 0;
  private tagStart = 0;

  constructor(private readonly text: string) {}

  /** Advance to the next element tag.  False when the document is exhausted. */
  next(): boolean {
    const text = this.text;
    let i = text.indexOf("<", this.index);
    // Skip the declaration, comments and processing instructions.
    while (i >= 0 && (text[i + 1] === "?" || text[i + 1] === "!")) {
      const end = text.indexOf(">", i);
      if (end < 0) return false;
      i = text.indexOf("<", end + 1);
    }
    if (i < 0) {
      this.index = text.length;
      return false;
    }
    const end = text.indexOf(">", i);
    if (end < 0) throw new XmlError("unterminated tag", i);

    this.tagStart = i;
    this.closing = text[i + 1] === "/";
    this.selfClosing = text[end - 1] === "/";
    const nameStart = i + (this.closing ? 2 : 1);
    let nameEnd = nameStart;
    while (nameEnd < end) {
      const c = text.charCodeAt(nameEnd);
      // space, tab, newline, carriage return, slash
      if (c === 32 || c === 9 || c === 10 || c === 13 || c === 47) break;
      nameEnd++;
    }
    this.name = text.slice(nameStart, nameEnd);
    this.attributeStart = nameEnd;
    this.attributeEnd = this.selfClosing ? end - 1 : end;
    this.index = end + 1;
    return true;
  }

  /** The current attribute's raw value, or undefined when it is absent. */
  attr(name: string): string | undefined {
    const text = this.text;
    let i = this.attributeStart;
    const end = this.attributeEnd;
    while (i < end) {
      // Attribute names are preceded by whitespace in everything OpenMM writes.
      const found = text.indexOf(name, i);
      if (found < 0 || found >= end) return undefined;
      const before = text.charCodeAt(found - 1);
      const after = found + name.length;
      const isWholeName =
        (before === 32 || before === 9 || before === 10 || before === 13) &&
        text.charCodeAt(after) === 61 /* = */ &&
        text.charCodeAt(after + 1) === 34 /* " */;
      if (isWholeName) {
        const valueStart = after + 2;
        const valueEnd = text.indexOf('"', valueStart);
        if (valueEnd < 0 || valueEnd > end) {
          throw new XmlError(`unterminated value for ${name}`, valueStart);
        }
        return text.slice(valueStart, valueEnd);
      }
      i = found + name.length;
    }
    return undefined;
  }

  /**
   * A required numeric attribute.
   *
   * OpenMM writes numbers like `.1522` and `-0`, both of which `Number()`
   * handles; an empty string is not, which is why that is checked.
   */
  num(name: string): number {
    const raw = this.attr(name);
    if (raw === undefined || raw === "") {
      throw new XmlError(`<${this.name}> has no ${name} attribute`, this.tagStart);
    }
    const value = Number(raw);
    if (Number.isNaN(value)) {
      throw new XmlError(`<${this.name}> ${name}="${raw}" is not a number`, this.tagStart);
    }
    return value;
  }

  int(name: string): number {
    const value = this.num(name);
    if (!Number.isInteger(value)) {
      throw new XmlError(`<${this.name}> ${name} is not an integer`, this.tagStart);
    }
    return value;
  }

  /** An optional numeric attribute. */
  numOr(name: string, fallback: number): number {
    const raw = this.attr(name);
    return raw === undefined || raw === "" ? fallback : Number(raw);
  }

  /** OpenMM writes booleans as "0" and "1". */
  flag(name: string, fallback = false): boolean {
    const raw = this.attr(name);
    if (raw === undefined) return fallback;
    return raw === "1" || raw === "true";
  }

  /** A required string attribute. */
  str(name: string): string {
    const raw = this.attr(name);
    if (raw === undefined) {
      throw new XmlError(`<${this.name}> has no ${name} attribute`, this.tagStart);
    }
    return raw;
  }

  /** Where the current tag starts, for error messages. */
  get offset(): number {
    return this.tagStart;
  }

  /**
   * Consume everything up to and including the closing tag of the element the
   * scanner is on, which must not be self-closing.
   */
  skipElement(): void {
    if (this.selfClosing) return;
    const name = this.name;
    let depth = 1;
    while (this.next()) {
      if (this.name !== name) continue;
      if (this.closing) {
        if (--depth === 0) return;
      } else if (!this.selfClosing) {
        depth++;
      }
    }
    throw new XmlError(`<${name}> is never closed`, this.offset);
  }
}
