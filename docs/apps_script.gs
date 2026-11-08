/**
 * SignSpeak survey -> Google Sheet.
 * Paste this whole file into Extensions > Apps Script of your Google Sheet, then Deploy > New deployment > Web app
 * (Execute as: Me, Who has access: Anyone). Copy the Web app URL into docs/config.js (SHEET_URL).
 */
var COLUMNS = ["submitted_at","id","src","seconds_in_app","roles_visited","survey_seconds","screen",
  "role","device","familiarity","usability","learnability","affordance","visibility","feedback","navigation",
  "accessibility","efficiency","aesthetics","clarity","recommend","improve","comment"];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Responses") ||
                SpreadsheetApp.getActiveSpreadsheet().insertSheet("Responses");
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(COLUMNS);
      sheet.setFrozenRows(1);
      sheet.getRange(1, 1, 1, COLUMNS.length).setFontWeight("bold").setBackground("#CDEBF3");
    }
    var idCol = COLUMNS.indexOf("id") + 1;
    if (sheet.getLastRow() > 1) {                                   // ignore duplicates (double click / retry)
      var ids = sheet.getRange(2, idCol, sheet.getLastRow() - 1, 1).getValues();
      for (var i = 0; i < ids.length; i++) if (ids[i][0] === data.id) return ok("duplicate");
    }
    sheet.appendRow(COLUMNS.map(function (c) { return clean(data[c]); }));
    return ok("saved");
  } catch (err) {
    return ok("error: " + err);
  } finally {
    lock.releaseLock();
  }
}

function doGet() { return ok("SignSpeak survey endpoint is working"); }

function clean(v) {                     // stop spreadsheet formulas typed by a participant ("=...", "+...")
  if (v === undefined || v === null) return "";
  v = String(v);
  return /^[=+\-@]/.test(v) ? "'" + v : v;
}
function ok(msg) { return ContentService.createTextOutput(msg); }
