//! Keeps the chat box out from under an auto-hiding Windows taskbar.
//!
//! When the taskbar is set to auto-hide, Windows gives applications the whole screen (the work
//! area equals the monitor), and the taskbar slides up *over* the bottom of the window whenever
//! the mouse reaches that edge -- exactly where the chat box is, so typing a message meant
//! fighting the taskbar (reported 2026-11-04). A normal (always-shown) taskbar is already excluded
//! from the work area, so maximized windows never reach it and nothing needs doing.
//!
//! `taskbar_inset` reports how far the auto-hidden taskbar would cover the window on each edge
//! (physical pixels); the UI pads that edge so its content always stays clear.

use serde::Serialize;

#[derive(Debug, Default, Clone, Copy, Serialize, PartialEq)]
pub struct Inset {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

/// A rectangle as (left, top, right, bottom), physical pixels.
type Rect = (i32, i32, i32, i32);

/// Pure so it can be tested: given the taskbar's edge (0 left, 1 top, 2 right, 3 bottom, as in
/// `ABE_*`), its thickness, the monitor it is on, and the window, how much of each window edge
/// the taskbar covers when it slides out. Zero unless the window actually reaches that edge of
/// the taskbar's monitor.
pub fn inset_for(edge: u32, thickness: u32, taskbar_monitor: Rect, window: Rect) -> Inset {
    let (ml, mt, mr, mb) = taskbar_monitor;
    let (wl, wt, wr, wb) = window;
    // The window must be on the taskbar's monitor at all.
    let overlaps = wl < mr && wr > ml && wt < mb && wb > mt;
    if !overlaps || thickness == 0 {
        return Inset::default();
    }
    // How much of the window lies within `thickness` of the monitor edge (window can overhang the
    // monitor slightly when maximized, so measure against the monitor, not the window).
    let covered = |window_edge_inside: i32| -> u32 { window_edge_inside.clamp(0, thickness as i32) as u32 };
    let mut out = Inset::default();
    match edge {
        3 => out.bottom = covered(wb.min(mb) - (mb - thickness as i32)),
        1 => out.top = covered((mt + thickness as i32) - wt.max(mt)),
        0 => out.left = covered((ml + thickness as i32) - wl.max(ml)),
        2 => out.right = covered(wr.min(mr) - (mr - thickness as i32)),
        _ => {}
    }
    out
}

#[cfg(windows)]
mod win {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct RECT {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }

    #[repr(C)]
    pub struct APPBARDATA {
        pub cb_size: u32,
        pub hwnd: isize,
        pub u_callback_message: u32,
        pub u_edge: u32,
        pub rc: RECT,
        pub l_param: isize,
    }

    #[link(name = "shell32")]
    extern "system" {
        fn SHAppBarMessage(dw_message: u32, p_data: *mut APPBARDATA) -> usize;
    }

    const ABM_GETSTATE: u32 = 4;
    const ABM_GETTASKBARPOS: u32 = 5;
    const ABS_AUTOHIDE: usize = 1;

    /// (edge, thickness, taskbar rect) when the taskbar is set to auto-hide; None otherwise.
    pub fn autohide_taskbar() -> Option<(u32, u32, RECT)> {
        let mut data = APPBARDATA {
            cb_size: std::mem::size_of::<APPBARDATA>() as u32,
            hwnd: 0,
            u_callback_message: 0,
            u_edge: 0,
            rc: RECT::default(),
            l_param: 0,
        };
        let state = unsafe { SHAppBarMessage(ABM_GETSTATE, &mut data) };
        if state & ABS_AUTOHIDE == 0 {
            return None;
        }
        if unsafe { SHAppBarMessage(ABM_GETTASKBARPOS, &mut data) } == 0 {
            return None;
        }
        let rc = data.rc;
        // The thickness is the short side; it is the same whether the bar is shown or tucked
        // away, while its position is not (a hidden bar sits mostly off-screen).
        let thickness = match data.u_edge {
            0 | 2 => rc.right - rc.left,
            _ => rc.bottom - rc.top,
        };
        if thickness <= 0 {
            return None;
        }
        Some((data.u_edge, thickness as u32, rc))
    }
}

/// How much of the window an auto-hiding taskbar would cover, per edge, in physical pixels.
/// All zeros when the taskbar is not auto-hiding, the window is elsewhere, or off Windows.
#[tauri::command]
pub fn taskbar_inset(window: tauri::WebviewWindow) -> Inset {
    #[cfg(windows)]
    {
        let Some((edge, thickness, rc)) = win::autohide_taskbar() else { return Inset::default() };
        let (Ok(pos), Ok(size)) = (window.outer_position(), window.outer_size()) else {
            return Inset::default();
        };
        let win_rect = (pos.x, pos.y, pos.x + size.width as i32, pos.y + size.height as i32);
        // The monitor that holds the taskbar: the one its centre line falls on.
        let (cx, cy) = ((rc.left + rc.right) / 2, (rc.top + rc.bottom) / 2);
        let monitor = window.available_monitors().ok().and_then(|ms| {
            ms.into_iter().find(|m| {
                let (p, s) = (m.position(), m.size());
                let (l, t, r, b) = (p.x, p.y, p.x + s.width as i32, p.y + s.height as i32);
                match edge {
                    0 | 2 => cy >= t && cy < b && (cx - l).abs() < s.width as i32 + thickness as i32,
                    _ => cx >= l && cx < r && (cy - t).abs() < s.height as i32 + thickness as i32,
                }
            })
        });
        let Some(m) = monitor else { return Inset::default() };
        let (p, s) = (m.position(), m.size());
        let mon_rect = (p.x, p.y, p.x + s.width as i32, p.y + s.height as i32);
        return inset_for(edge, thickness, mon_rect, win_rect);
    }
    #[cfg(not(windows))]
    {
        let _ = window;
        Inset::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MON: Rect = (0, 0, 1920, 1200);

    #[test]
    fn maximized_window_is_covered_by_a_bottom_taskbar() {
        // Maximized windows overhang the monitor by a few pixels on Windows.
        let win = (-11, -11, 1931, 1211);
        assert_eq!(inset_for(3, 72, MON, win), Inset { bottom: 72, ..Default::default() });
    }

    #[test]
    fn window_that_stops_short_of_the_taskbar_is_not_covered() {
        let win = (100, 100, 900, 800);
        assert_eq!(inset_for(3, 72, MON, win), Inset::default());
    }

    #[test]
    fn partly_covered_window_gets_only_the_covered_part() {
        let win = (100, 100, 900, 1160); // bottom 40 px into the 72 px strip
        assert_eq!(inset_for(3, 72, MON, win).bottom, 32);
    }

    #[test]
    fn other_edges_and_other_monitors() {
        assert_eq!(inset_for(1, 60, MON, (0, 0, 1920, 1200)).top, 60);
        assert_eq!(inset_for(0, 60, MON, (0, 0, 1920, 1200)).left, 60);
        assert_eq!(inset_for(2, 60, MON, (0, 0, 1920, 1200)).right, 60);
        // window on a second monitor to the right: untouched
        assert_eq!(inset_for(3, 72, MON, (1920, 0, 3840, 1200)), Inset::default());
        assert_eq!(inset_for(3, 0, MON, (0, 0, 1920, 1200)), Inset::default());
    }
}
